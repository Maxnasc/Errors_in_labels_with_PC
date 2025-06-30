import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from ocpc_py import OneClassPC
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import os
import pickle

from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


class PC_LabelCorrector:
    def __init__(self, path: str, detect_outlier_with_ocpc = False,  k_max = 3, alfa = 0.1545514066693123, lamda = 0.946236808357798, close = False, buffer = 1000, f = 0.6246253156643677, 
                 outlier_rate = 0.1):
        """
        Initializes the LabelCorrector with attributes to store state.
        """
        # config 01: k_max = 18, alfa = 0.7135, lamda = 1.1339, close = False, buffer = 1000, f = 0.593, outlier_rate = 0.1
        # config 02: k_max = 5, alfa = 0.7135, lamda = 1.1339, close = False, buffer = 1000, f = 0.593, outlier_rate = 0.1
        # config 03: k_max = 5, alfa = 0.7135, lamda = 1.1339, close = False, buffer = 1000, f = 1, outlier_rate = 0.1
        # config 04: k_max = 5, alfa = 0.7135, lamda = 1.1339, close = False, buffer = 1000, f = 0.593, outlier_rate = 0.1
        # config 05: k_max = 5, alfa = 0.7135, lamda = 1, close = False, buffer = 1000, f = 1, outlier_rate = 0.1
        # config 06: k_max = 5, alfa = 0.7135, lamda = 1.1339, close = False, buffer = 1000, f = 1, outlier_rate = 0.1
        # config 07: k_max = 10, alfa = 0.23264431253985635, lamda = 0.49746871412451305, close = False, buffer = 1000, f = 1, outlier_rate = 0.021446327968908222
        # config 09: k_max = 10, alfa = 0.6186, lamda = 0.5630, close = False, buffer = 1000, f = 1.2366, outlier_rate = 0.1

        self.X_separated = None
        self.indexes_to_swap = None
        self.Y_adjusted = None
        self.metrics = None
        self.detect_outlier_with_ocpc = detect_outlier_with_ocpc

        # parameters
        self.contamination = None
        self.k_max = k_max
        self.alfa  = alfa
        self.lamda = lamda
        self.buffer = buffer
        self.close = close
        self.f = f
        self.outlier_rate = outlier_rate
        
        self.path = path

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
        for k in np.unique(Y):
            classe_ = str(int(k))
            x_separated[classe_] = X[Y == int(k)]
            del classe_
        ##
        
        # for i in range(len(X)):
        #     if Y[i] not in x_separated:
        #         x_separated[Y[i]] = {"X": np.ndarray([])}
        #     x_separated[Y[i]]["X"].append(np.array(X[i]))

        return x_separated

    def _separate_X_in_inliers_and_outliers(self, x_separated: dict) -> dict:
        """
        Identifies inliers and outliers for each class using LOF.

        Args:
            x_separated: Dictionary with data separated by class

        Returns:
            Dictionary with identified inliers and outliers
        """
        result = {}
        for class_label, X in x_separated.items():
            if self.detect_outlier_with_ocpc:
                preditc, scores = self._detect_outliers_ocpc(np.array(X))
            else:
                preditc, scores = self._detect_outliers_lof(X)
            # X["x_inliers"] = np.array(
            #     [x for i, x in enumerate(X) if preditc[i] == 1]
            # )
            result[class_label] = {}
            result[class_label]["x_inliers"] = X[preditc==1]
            result[class_label]["x_outliers"] = X[preditc==-1]
            result[class_label]['predict'] = preditc
        return result

    # def _detect_outliers_lof(self, X):
    #     """
    #     Applies the Local Outlier Factor (LOF) to detect outliers.

    #     Args:
    #         X: Data for outlier detection

    #     Returns:
    #         Tuple with predictions and scores
    #     """
    #     lof = LocalOutlierFactor(
    #         n_neighbors=int(len(X)/2), contamination=self.contamination
    #     )
    #     y_pred = lof.fit_predict(X)
    #     scores = lof.negative_outlier_factor_
        
    #     # Apenas para dados 2D
    #     if X.shape[1] == 2:
    #         xx, yy = np.meshgrid(
    #             np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
    #             np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100),
    #         )
    #         grid = np.c_[xx.ravel(), yy.ravel()]
    #         Z = lof.decision_function(grid)  # <- Aqui funciona
    #         Z = Z.reshape(xx.shape)

    #         plt.figure(figsize=(8, 6))
    #         plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 50), cmap=plt.cm.RdBu_r)
    #         plt.colorbar(label='LOF Decision Function')
    #         plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap=plt.cm.coolwarm, edgecolors='k')
    #         plt.title("LOF Decision Boundary")
    #         plt.xlabel("Feature 1")
    #         plt.ylabel("Feature 2")
    #         plt.show()
        
    #     return y_pred, scores
    
    def _detect_outliers_lof(self, X):
        """
        Detecta outliers com LOF e plota as regiões de decisão.
        Pressupõe X com exatamente 2 features.
        """
        # 1) LOF preparado para novelty detection
        lof = LocalOutlierFactor(
            n_neighbors=int(len(X) / 2),
            contamination=self.contamination,
            novelty=True          # <- ponto crucial
        )
        lof.fit(X)                # 2) apenas fit

        # 3) Predição/score para os próprios dados (opcional – só para colorir o scatter)
        y_pred = lof.predict(X)                 # agora disponível
        scores  = lof.negative_outlier_factor_  # sempre disponível para treino

        # --- Plot da malha (grid) ---
        # if X.shape[1] == 2:                     # só faz sentido em 2D
        #     # 4) Gera grid “novo” (não visto no fit)
        #     xx, yy = np.meshgrid(
        #         np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 300),
        #         np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 300)
        #     )
        #     grid = np.c_[xx.ravel(), yy.ravel()]
        #     # Usa decision_function no grid
        #     Z = lof.decision_function(grid)
        #     Z = Z.reshape(xx.shape)

        #     plt.figure(figsize=(8, 6))
        #     # Quanto maior Z, mais “normal” é o ponto
        #     cs = plt.contourf(xx, yy, Z,
        #                     levels=np.linspace(Z.min(), Z.max(), 50),
        #                     cmap=plt.cm.RdBu_r)
        #     plt.colorbar(cs, label="decision_function")
        #     # pinta inliers/outliers do conjunto original
        #     plt.scatter(X[:, 0], X[:, 1],
        #                 c=y_pred,           # 1 = inlier, -1 = outlier
        #                 cmap=plt.cm.coolwarm, edgecolors="k")
        #     plt.title("Fronteira de decisão do LOF")
        #     plt.xlabel("Feature 1")
        #     plt.ylabel("Feature 2")
        #     plt.tight_layout()
        #     plt.savefig(f'tests/{self.path}/imagens/regiao de decisao LOF.png')
            
            # plt.show()

        return y_pred, scores
    
    def _detect_outliers_ocpc(self, X):
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
        ocpc = OneClassPC(k_max = self.k_max, alfa = self.alfa, lamda = self.lamda, close = self.close, buffer = self.buffer, f = self.f, outlier_rate = self.outlier_rate)
        
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
        clf = OneClassPC(k_max = self.k_max, alfa = self.alfa, lamda = self.lamda, close = self.close, buffer = self.buffer, f = self.f, outlier_rate = self.outlier_rate)
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
        a = x_separated.copy()
        result = x_separated.copy()
        for class_label, X in a.items():
            result[class_label]["curve"] = self._get_OneClass_curve(X.get("x_inliers"))
        #     # Plotar a curva com os dados de inliers e outliers indicados
        #     fig, ax = plt.subplots()
        #     x_inliers = X.get("x_inliers")
        #     x_outliers = X.get("x_outliers")
        #     if x_inliers is not None and len(x_inliers) > 0:
        #         if x_inliers.ndim == 2 and x_inliers.shape[1] >= 2:
        #             ax.scatter(x_inliers[:, 0], x_inliers[:, 1], marker='o', label='Inliers')
        #         else:
        #             ax.scatter(np.arange(len(x_inliers)), x_inliers, marker='o', label='Inliers')
        #     if x_outliers is not None and len(x_outliers) > 0:
        #         if x_outliers.ndim == 2 and x_outliers.shape[1] >= 2:
        #             ax.scatter(x_outliers[:, 0], x_outliers[:, 1], marker='*', label='Outliers')
        #         else:
        #             ax.scatter(np.arange(len(x_outliers)), x_outliers, marker='*', label='Outliers')
        #     result[class_label]["curve"].plot_curve(ax)
        #     plt.savefig(f'tests/{self.path}/imagens/classe_{class_label}.png')
            
        # plt.show()
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
                if len(x["x_outliers"]) > 0:
                    _, dists = curve.map_to_arcl(x["x_outliers"])
                else:
                    dists = []
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
    
    def save_outliers(self):
        outliers = []
        for classe, valores_classes in self.X_separated.items():
            outliers.extend(valores_classes.get('predict'))
        
        # if self.detect_outlier_with_ocpc:
        #     caminho = f'tests/{self.path}/outliers_ocpc.json'
        # else:
        #     caminho = f'tests/{self.path}/outliers_lof.json'
            
        # with open(caminho, "w") as f:
        #     json.dump([int(o) for o in outliers], f, indent=4)

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

        # # Step 0: Normalizing X
        # scaler = StandardScaler()
        # X_scaled = scaler.fit_transform(X)

        f = open('x.pkl', "wb")
        pickle.dump(X, f)
        f.close()
        
        f = open('y.pkl', "wb")
        pickle.dump(Y, f)
        f.close()
        
        # Step 01: Separate X and Y according to each class
        self.X_separated = self._separate_for_each_class(X=X, Y=Y)
        
        for i, x in self.X_separated.items():
            plt.scatter(x[:,0], x[:,1])
            plt.title(f'X classe {i}')

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

        self.save_outliers()
        
        return self.Y_adjusted

    def save_metrics_to_json_file(self, path: str):
        # Save results to a JSON file
        if ".json" not in path:
            path = path + ".json"

        with open(path, "w") as f:
            json.dump(self.metrics, f, indent=4)

        print(f"Results saved to {path}")


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
    
    # TODO: Preciso tentar corrigir os rótulos de imagens usando o PC_LabelCorrector
    
