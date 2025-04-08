import numpy as np
import pandas as pd
from ocpc_py import OneClassPC
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor

import numpy as np
import pandas as pd
from ocpc_py import OneClassPC
from sklearn.neighbors import LocalOutlierFactor


class LabelCorrector:
    def __init__(self):
        """
        Initializes the LabelCorrector with attributes to store state.
        """
        self.X_separated = None
        self.indices_to_swap = None
        self.Y_adjusted = None

    def _separate_for_each_class(self, X: np.array, Y: np.array) -> dict:
        """
        Separates the data X into different classes based on Y.

        Args:
            X: Numpy array with feature data
            Y: Numpy array with labels

        Returns:
            Dictionary with data separated by class
        """
        x_separated = {}

        for i in range(len(X)):
            if Y[i] not in x_separated:
                x_separated[Y[i]] = {"X": []}
            x_separated[Y[i]]["X"].append(np.array(X[i]))

        return x_separated

    def _separate_X_in_inliers_and_outliers(self, x_separated: dict) -> dict:
        """
        Identifies inliers and outliers for each class using LOF.

        Args:
            x_separated: Dictionary with data separated by class

        Returns:
            Dictionary with identified inliers and outliers
        """
        result = x_separated.copy()
        for class_label, X in x_separated.items():
            preditc, scores = self._detect_outliers_lof(
                X["X"], n_neighbors=int(len(X["X"]) / 2), contamination="auto"
            )
            X["x_inliers"] = np.array(
                [x for i, x in enumerate(X["X"]) if preditc[i] == 1]
            )
            X["x_outliers"] = np.array(
                [x for i, x in enumerate(X["X"]) if preditc[i] == -1]
            )
        return result

    def _detect_outliers_lof(self, X, n_neighbors=50, contamination="auto"):
        """
        Applies the Local Outlier Factor (LOF) to detect outliers.

        Args:
            X: Data for outlier detection
            n_neighbors: Number of neighbors for LOF
            contamination: Contamination parameter for LOF

        Returns:
            Tuple with predictions and scores
        """
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        y_pred = lof.fit_predict(X)
        scores = lof.negative_outlier_factor_
        return y_pred, scores

    def _get_OneClass_curve(self, x_inlier_train):
        """
        Gets the OneClassPC curve for the inliers.

        Args:
            x_inlier_train: Inlier data to train the model

        Returns:
            OneClassPC curve
        """
        clf = OneClassPC()
        clf.fit(x_inlier_train)
        return clf.curve

    def _get_OneClassCurves(self, x_separated: dict) -> dict:
        """
        Gets the OneClassPC curves for each class.

        Args:
            x_separated: Dictionary with data separated by class

        Returns:
            Dictionary with curves for each class
        """
        result = x_separated.copy()
        for class_label, X in result.items():
            X['curve'] = self._get_OneClass_curve(X.get('x_inliers'))
        return result

    def _identify_indices_to_adjust(self, x_outlier_labeled, X):
        """
        Identifies indices that need to be adjusted.

        Args:
            x_outlier_labeled: Outliers with adjusted labels
            X: Original data

        Returns:
            Dictionary with indices and new labels
        """
        x_adjusted_values = x_outlier_labeled.iloc[:, :-1].values
        x_adjusted_labels = x_outlier_labeled.iloc[:, -1].values
        
        X_values = X if isinstance(X, np.ndarray) else X.iloc.values
        
        matching_info = []
        for idx, row in enumerate(x_adjusted_values):
            matches = np.where((X_values == row).all(axis=1))[0]
            for match_idx in matches:
                matching_info.append({
                    'index': int(match_idx),
                    'label': x_adjusted_labels[idx]
                })
        
        unique_matches = {}
        for info in matching_info:
            if info['index'] not in unique_matches:
                unique_matches[info['index']] = info['label']
        
        return {
            'indices': sorted(unique_matches.keys()),
            'labels': [unique_matches[idx] for idx in sorted(unique_matches.keys())]
        }

    def _calculate_distances(self, x_separated: dict, X: np.array):
        """
        Calculates distances of outliers to each curve.

        Args:
            x_separated: Dictionary with data separated by class
            X: Original data

        Returns:
            Tuple with (updated result, indices to swap)
        """
        result = x_separated.copy()
        indices_to_swap = {
            'indices': [],
            'labels': []
        }
        
        curves_labeled = {class_label: X.get('curve') for class_label, X in x_separated.items()}
        
        for class_label, x in result.items():
            distances_df = pd.DataFrame()
            
            for label, curve in curves_labeled.items():
                _, dists = curve.map_to_arcl(x['x_outliers'])
                distances_df[label] = dists
    
            adjusted_labels = distances_df.idxmin(axis=1)
            x_outlier_labeled = pd.DataFrame(x['x_outliers'])
            x_outlier_labeled['adjusted_labels'] = adjusted_labels
            
            aux = self._identify_indices_to_adjust(x_outlier_labeled, X)
            x['indices_to_swap'] = aux
            for key, content in aux.items():
                indices_to_swap[key].extend(content)
        
        return result, indices_to_swap

    def _change_labels_on_Y(self, indices_to_swap: dict, Y: np.array) -> np.array:
        """
        Adjusts the labels of Y according to the identified indices.

        Args:
            indices_to_swap: Dictionary with indices and new labels
            Y: Numpy array with original labels

        Returns:
            Numpy array with adjusted labels
        """
        y_adjusted = Y.copy()
        for i, indice in enumerate(indices_to_swap['indices']):
            y_adjusted[indice] = indices_to_swap['labels'][i]
        return y_adjusted

    def run(self, X: np.array, Y: np.array) -> np.array:
        """
        Executes the complete label correction pipeline.

        Args:
            X: Numpy array with feature data
            Y: Numpy array with labels

        Returns:
            Numpy array with adjusted labels
        """
        # Step 01: Separate X and Y according to each class
        self.X_separated = self._separate_for_each_class(X=X, Y=Y)

        # Step 02: Find the inliers and outliers
        self.X_separated = self._separate_X_in_inliers_and_outliers(x_separated=self.X_separated)
        
        # Step 03: Get the curves for each class with inliers and outliers
        self.X_separated = self._get_OneClassCurves(x_separated=self.X_separated)
        
        # Step 04 and 05: Calculate distances and identify indices to swap
        self.X_separated, self.indices_to_swap = self._calculate_distances(
            x_separated=self.X_separated, X=X
        )
        
        # Step 06: Adjust the labels
        self.Y_adjusted = self._change_labels_on_Y(
            indices_to_swap=self.indices_to_swap, Y=Y
        )
        
        return self.Y_adjusted
    
# >>>>>>>>>>>>>>>>>>>>> TEST ONLY <<<<<<<<<<<<<<<<<<<<<<<<<<

def get_dataset_with_error(data, erro_proposto):

    def alterar_rotulos(Y, percentual, random_state=None):
        """
        Alters the labels of Y by a given percentage.

        Args:
            Y: Original labels
            percentual: Percentage of labels to alter
            random_state: Seed for reproducibility

        Returns:
            Altered labels
        """
        np.random.seed(random_state)  # For reproducibility
        Y_altered = Y.copy()
        classes = np.unique(Y)

        for classe in classes:
            class_indices = np.where(Y == classe)[0]  # Get indices of the class
            n_to_alter = int(len(class_indices) * percentual)
            chosen_indices = np.random.choice(
                class_indices, n_to_alter, replace=False
            )

            # Choose new random labels, different from the original
            for idx in chosen_indices:
                new_classes = np.setdiff1d(classes, Y[idx])  # Avoid the same label
                Y_altered[idx] = np.random.choice(new_classes)

        return Y_altered

    X = data.data  # Features
    Y_original = data.target  # Labels

    Y = alterar_rotulos(Y_original, erro_proposto)

    # Rebuild data_with_error
    data_with_error = {"data": data.data, "target": Y, "Y_original": Y_original}

    return data_with_error


if __name__ == "__main__":
    iris = load_iris()
    erro_proposto = 0.1
    results = {}
    iris_with_error = get_dataset_with_error(iris, erro_proposto)

    labels_wrong_before_adjustments = 0
    for i, y in enumerate(iris_with_error['target']):
        if y != iris.target[i]:
            labels_wrong_before_adjustments += 1

    lc = LabelCorrector()
    Y_adjusted = lc.run(X=iris_with_error["data"], Y=iris_with_error["target"])
    
    labels_wrong_after_adjustments = 0
    for i, y in enumerate(Y_adjusted):
        if y != iris.target[i]:
            labels_wrong_after_adjustments += 1
    
    print(f"labels_wrong_before_adjustments {labels_wrong_before_adjustments}")
    print(f"labels_wrong_after_adjustments {labels_wrong_after_adjustments}")
    print(f"Erro diminuido de {erro_proposto*100}% para {round((labels_wrong_after_adjustments/len(iris_with_error['data'])), 2)*100}%")
    a = 1
    
