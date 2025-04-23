import json
import numpy as np
import pandas as pd
from ocpc_py import OneClassPC
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
import os

from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


class PC_LabelCorrector:
    def __init__(self, detect_outlier_with_ocpc = True):
        """
        Initializes the LabelCorrector with attributes to store state.
        """
        self.X_separated = None
        self.indexes_to_swap = None
        self.Y_adjusted = None
        self.metrics = None
        self.detect_outlier_with_ocpc = detect_outlier_with_ocpc

        # parameters
        self.contamination = None

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
            if self.detect_outlier_with_ocpc:
                preditc, scores = self._detect_outliers_ocpc(np.array(X["X"]))
            else:
                preditc, scores = self._detect_outliers_lof(X["X"])
            X["x_inliers"] = np.array(
                [x for i, x in enumerate(X["X"]) if preditc[i] == 1]
            )
            X["x_outliers"] = np.array(
                [x for i, x in enumerate(X["X"]) if preditc[i] == -1]
            )
        return result

    def _detect_outliers_lof(self, X):
        """
        Applies the Local Outlier Factor (LOF) to detect outliers.

        Args:
            X: Data for outlier detection

        Returns:
            Tuple with predictions and scores
        """
        lof = LocalOutlierFactor(
            n_neighbors=int(len(X)/2), contamination=self.contamination
        )
        y_pred = lof.fit_predict(X)
        scores = lof.negative_outlier_factor_
        return y_pred, scores
    
    def _detect_outliers_ocpc(self, X, n_segments=10, contamination=0.1):
        """
        Applies the One-Class Principal Curve (OCPC) to detect outliers.

        Args:
            X (array-like): data for outlier detection.
            n_segments (int): number of segments for the principal curve.
            contamination (float): expected proportion of outliers (OR).

        Returns:
            outlier_indices (np.ndarray): indices of detected outliers.
            scores (np.ndarray): distances from each point to the principal curve.
        """
        # Instantiate the OCPC classifier
        ocpc = OneClassPC(k_max=n_segments, outlier_rate=contamination)
        
        # Fit the model on the data
        ocpc.fit(X)
        
        # Compute scores (Euclidean distances to the principal curve)
        scores = ocpc.fit(X)
        
        # Predict labels: +1 for inliers, -1 for outliers
        y_pred = ocpc.predict(X)
        
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
            X["curve"] = self._get_OneClass_curve(X.get("x_inliers"))
        return result

    def _identify_indexes_to_adjust(self, x_outlier_labeled, X):
        """
        Identifies indexes that need to be adjusted.

        Args:
            x_outlier_labeled: Outliers with adjusted labels
            X: Original data

        Returns:
            Dictionary with indexes and new labels
        """
        x_adjusted_values = x_outlier_labeled.iloc[:, :-1].values
        x_adjusted_labels = x_outlier_labeled.iloc[:, -1].values

        X_values = X if isinstance(X, np.ndarray) else X.iloc.values

        matching_info = []
        for idx, row in enumerate(x_adjusted_values):
            matches = np.where((X_values == row).all(axis=1))[0]
            for match_idx in matches:
                matching_info.append(
                    {"index": int(match_idx), "label": x_adjusted_labels[idx]}
                )

        unique_matches = {}
        for info in matching_info:
            if info["index"] not in unique_matches:
                unique_matches[info["index"]] = info["label"]

        return {
            "indexes": sorted(unique_matches.keys()),
            "labels": [unique_matches[idx] for idx in sorted(unique_matches.keys())],
        }

    def _calculate_distances(self, x_separated: dict, X: np.array):
        """
        Calculates distances of outliers to each curve.

        Args:
            x_separated: Dictionary with data separated by class
            X: Original data

        Returns:
            Tuple with (updated result, indexes to swap)
        """
        result = x_separated.copy()
        indexes_to_swap = {"indexes": [], "labels": []}

        curves_labeled = {
            class_label: X.get("curve") for class_label, X in x_separated.items()
        }

        for class_label, x in result.items():
            distances_df = pd.DataFrame()

            for label, curve in curves_labeled.items():
                _, dists = curve.map_to_arcl(x["x_outliers"])
                distances_df[label] = dists

            adjusted_labels = distances_df.idxmin(axis=1)
            x_outlier_labeled = pd.DataFrame(x["x_outliers"])
            x_outlier_labeled["adjusted_labels"] = adjusted_labels

            aux = self._identify_indexes_to_adjust(x_outlier_labeled, X)
            x["indexes_to_swap"] = aux
            for key, content in aux.items():
                indexes_to_swap[key].extend(content)

        return result, indexes_to_swap

    def _change_labels_on_Y(self, indexes_to_swap: dict, Y: np.array) -> np.array:
        """
        Adjusts the labels of Y according to the identified indexes.

        Args:
            indexes_to_swap: Dictionary with indexes and new labels
            Y: Numpy array with original labels

        Returns:
            Numpy array with adjusted labels
        """
        y_adjusted = Y.copy()
        for i, indice in enumerate(indexes_to_swap["indexes"]):
            y_adjusted[indice] = indexes_to_swap["labels"][i]
        return y_adjusted

    def _mount_metrics(self):
        # saves the label corrector metrics into a json file
        metrics = {
            "number of possibly incorrect labels": 0,
            "number of labels fixed": 0,
            "corrected label indexes": [],
            "corrected labels": [],
            "original labels": [],
            "original error rate": 0.0,
            "error rate after correction": 0.0,
        }

        # Getting the corrected label indexes
        metrics["corrected label indexes"] = list(self.indexes_to_swap.get("indexes"))

        # Getting the corrected labels
        metrics["corrected labels"] = [
            int(label) for label in self.indexes_to_swap.get("labels")
        ]

        # Getting the original labels
        metrics["original labels"] = [
            int(label)
            for i, label in enumerate(self.Y_original)
            if i in self.indexes_to_swap.get("indexes")
        ]

        # Getting the number of possibly incorrect labels
        metrics["number of possibly incorrect labels"] = len(
            self.indexes_to_swap.get("labels")
        )

        # Getting the number of labels fixed
        metrics["number of labels fixed"] = len(
            [
                label
                for i, label in enumerate(self.indexes_to_swap.get("labels"))
                if metrics["original labels"][i] != label
            ]
        )

        # Getting the original error rate
        metrics["original error rate"] = round(
            (metrics.get("number of possibly incorrect labels") / len(self.Y_original)),
            4,
        )

        # Getting the error rate after correction
        metrics["error rate after correction"] = round(
            (abs(metrics["number of possibly incorrect labels"]-metrics.get("number of labels fixed")) / len(self.Y_original)), 4
        )

        return metrics

    def run(
        self, X: np.array, Y: np.array, contamination="auto"
    ) -> np.array:
        """
        Executes the complete label correction pipeline.

        Args:
            X: Numpy array with feature data
            Y: Numpy array with labels
            n_neighbors: Parameter of LOF. Number of neighbors to use by default for kneighbors queries.  Possible values: int natural numbers
            contamination: Parameter of LOF. The amount of contamination of the data set, i.e. the proportion of outliers in the data set. Shoud be on [0, 0.5] range.

        Returns:
            Numpy array with adjusted labels
        """
        self.Y_original = Y.copy()

        # Verifing contamination
        if contamination == "auto":
            self.contamination = contamination
        elif (
            (type(contamination) == float)
            and (contamination <= 0.5)
            and (contamination >= 0)
        ):
            self.contamination = contamination
        else:
            raise Exception(
                "contamination parameter should be in the [0, 0.5] range or 'auto'. Please try again with a different value for contamination"
            )

        # Step 01: Separate X and Y according to each class
        self.X_separated = self._separate_for_each_class(X=X, Y=Y)

        # Step 02: Find the inliers and outliers
        self.X_separated = self._separate_X_in_inliers_and_outliers(
            x_separated=self.X_separated
        )
        
        # separated = self._detect_outliers_ocpc(self, self.X_separated)

        # Step 03: Get the curves for each class with inliers and outliers
        self.X_separated = self._get_OneClassCurves(x_separated=self.X_separated)

        # Step 04 and 05: Calculate distances and identify indexes to swap
        self.X_separated, self.indexes_to_swap = self._calculate_distances(
            x_separated=self.X_separated, X=X
        )

        # Step 06: Adjust the labels
        self.Y_adjusted = self._change_labels_on_Y(
            indexes_to_swap=self.indexes_to_swap, Y=Y
        )

        self.metrics = self._mount_metrics()

        return self.Y_adjusted

    def save_metrics_to_json_file(self, path: str):
        # Save results to a JSON file
        if ".json" not in path:
            path = path + ".json"

        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        print(f"Results saved to {path}")


# >>>>>>>>>>>>>>>>>>>>> TEST ONLY <<<<<<<<<<<<<<<<<<<<<<<<<<


# def get_dataset_with_error(data, erro_proposto):

#     def alterar_rotulos(Y, percentual, random_state=None):
#         """
#         Alters the labels of Y by a given percentage.

#         Args:
#             Y: Original labels
#             percentual: Percentage of labels to alter
#             random_state: Seed for reproducibility

#         Returns:
#             Altered labels
#         """
#         np.random.seed(random_state)  # For reproducibility
#         Y_altered = Y.copy()
#         classes = np.unique(Y)

#         for classe in classes:
#             class_indexes = np.where(Y == classe)[0]  # Get indexes of the class
#             n_to_alter = int(len(class_indexes) * percentual)
#             chosen_indexes = np.random.choice(class_indexes, n_to_alter, replace=False)

#             # Choose new random labels, different from the original
#             for idx in chosen_indexes:
#                 new_classes = np.setdiff1d(classes, Y[idx])  # Avoid the same label
#                 Y_altered[idx] = np.random.choice(new_classes)

#         return Y_altered

#     X = data.data  # Features
#     Y_original = data.target  # Labels

#     Y = alterar_rotulos(Y_original, erro_proposto)

#     # Rebuild data_with_error
#     data_with_error = {"data": data.data, "target": Y, "Y_original": Y_original}

#     return data_with_error


if __name__ == "__main__":
    iris = load_iris()
    erro_proposto = 0.1
    results = {}
    iris_with_error = get_dataset_with_error(iris.data, iris.target, erro_proposto)

    labels_wrong_before_adjustments = 0
    for i, y in enumerate(iris_with_error['target']):
        if y != iris.target[i]:
            labels_wrong_before_adjustments += 1

    # Comparação com o detector de erros com PC
    
    lc = PC_LabelCorrector(detect_outlier_with_ocpc=True)
    Y_adjusted = lc.run(X=iris_with_error["data"], Y=iris_with_error["target"])
    
    # Comparação com o CL
    CL_issues = get_CL_label_correction(iris_with_error["data"], iris_with_error["target"], iris.target)
    
    # TODO: Gerar um arquivo json com as métricas do CL e do PC_labelCorrector para comparação
    # lc.save_metrics_to_json_file(path='tests/load_iris/results_LabelCorrector_load_iris')
    
    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues
    
    path='ocpc_detection'
    # for metric, value in metrics.items():
#        print(f"{metric}: {value}")
        
    save_metrics_to_json_file(path=path, metrics=metrics)
    
    # LOF detection
    
    lc = PC_LabelCorrector(detect_outlier_with_ocpc=False)
    Y_adjusted = lc.run(X=iris_with_error["data"], Y=iris_with_error["target"])
    
    # Comparação com o CL
    CL_issues = get_CL_label_correction(iris_with_error["data"], iris_with_error["target"], iris.target)
    
    # TODO: Gerar um arquivo json com as métricas do CL e do PC_labelCorrector para comparação
    # lc.save_metrics_to_json_file(path='tests/load_iris/results_LabelCorrector_load_iris')
    
    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues
    
    path='lof_detection'
    # for metric, value in metrics.items():
#        print(f"{metric}: {value}")
        
    save_metrics_to_json_file(path=path, metrics=metrics)
    
    # TODO: Preciso implementar a identificação de outliers com o ocpc
    # TODO: Preciso tentar corrigir os rótulos de imagens usando o PC_LabelCorrector
    
