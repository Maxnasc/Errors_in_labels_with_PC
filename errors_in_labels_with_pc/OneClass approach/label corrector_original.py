import numpy as np
import pandas as pd
from ocpc_py import OneClassPC
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor

class LabelCorrector:
    def __init__(self, n_neighbors='auto', contamination='auto'):
        """
        Initializes the class with parameters for the number of neighbors and contamination level.
        Parameters:
        -----------
        n_neighbors : int or str, default='auto'
            The number of neighbors to consider. Must be a positive integer or the string 'auto'.
            If 'auto', the number of neighbors will be determined automatically.
        contamination : str, default='auto'
            The amount of contamination in the dataset. Currently set to 'auto'.
        Attributes:
        -----------
        n_neighbors : int or str
            Stores the value of the number of neighbors.
        contamination : str
            Stores the contamination level.
        Y_adjusted : numpy.ndarray
            An empty array initialized to store adjusted labels.
        X_separated : dict
            A dictionary initialized to store separated data.
        """
        
        # Verify if n_neighbors is int number or 'auto' string
        if not (isinstance(n_neighbors, int) and n_neighbors > 0) and n_neighbors != "auto":
            raise ValueError("n_neighbors must be a positive integer or the string 'auto'.")

        self.n_neighbors = n_neighbors
        self.contamination = contamination
        
        self.Y_adjusted = np.array([])  # Inicializa como array vazio
        self.X_separated = {}  # Dicionário para armazenar dados separados
        
        self.indices_to_swap = {}

    def run(self, X: np.array, Y: np.array):
        if self.n_neighbors == 'auto':
            self.n_neighbors = int(len(X) / 2)
            
        # Step 01: Separate X and Y according to each class
        self.separate_for_each_class(X=X, Y=Y)

        # Step 02: Find the inliers and outliers
        self.separate_X_in_inliers_and_outliers()
        
        # Step 03: Get the curves for each class with inliers and outliers
        self.get_OneClassCurves()
        
        # Step 04 and 05: Calculate the distance from each outlier to each curve including the original curve
        self.calculate_distances(X=X)
        
        # Step 06: Reassemble Y according to the adjustments made
        self.change_labels_on_Y(indices_to_swap=self.indices_to_swap, Y=Y)
        
        return self.Y_adjusted

    def separate_for_each_class(self, X: np.array, Y: np.array):
        # Dictionary of X values separated into classes
        x_separated = {}

        # Loop through the X values ​​by separating them
        for i in range(len(X)):
            if Y[i] not in x_separated:
                x_separated[Y[i]] = {"X": []}
            x_separated[Y[i]]["X"].append(np.array(X[i]))

        self.X_separated = x_separated

    def separate_X_in_inliers_and_outliers(self):
        for class_label, X in self.X_separated.items():
            # Obtain outliers with LOF
            preditc, scores = self.detect_outliers_lof(
                X["X"]
            )
            # Identify inliers on X
            X["x_inliers"] = np.array(
                [x for i, x in enumerate(X["X"]) if preditc[i] == 1]
            )
            # Identify outliers on X
            X["x_outliers"] = np.array(
                [x for i, x in enumerate(X["X"]) if preditc[i] == -1]
            )

    def detect_outliers_lof(self, X):
        """Aplica o Local Outlier Factor (LOF) para detectar outliers."""
        lof = LocalOutlierFactor(n_neighbors=self.n_neighbors, contamination=self.contamination)
        y_pred = lof.fit_predict(X)
        scores = lof.negative_outlier_factor_
        return y_pred, scores

    def get_OneClass_curve(self, x_inlier_train):
        clf = OneClassPC()
        # removing the last column (target column)
        clf.fit(x_inlier_train)

        return clf.curve

    def get_OneClassCurves(self):
        '''Obter as curvas para cada classe dentro de x_separated'''
        for class_label, X in self.X_separated.items():
            X['curve'] = self.get_OneClass_curve(X.get('x_inliers')) # Obtain the curve for inliers data

    def identify_indices_to_adjust(self, x_outlier_labeled, X):
        # 1. Extrai os valores numéricos e os labels
        x_adjusted_values = x_outlier_labeled.iloc[:, :-1].values
        x_adjusted_labels = x_outlier_labeled.iloc[:, -1].values  # Pega a coluna adjusted_labels
        
        # 2. Converte X para array numpy
        X_values = X if isinstance(X, np.ndarray) else X.iloc.values
        
        # 3. Encontra os índices de correspondência e mapeia aos labels
        matching_info = []
        for idx, row in enumerate(x_adjusted_values):
            matches = np.where((X_values == row).all(axis=1))[0]
            for match_idx in matches:
                matching_info.append({
                    'index': int(match_idx),  # Índice em X
                    'label': x_adjusted_labels[idx]  # Label correspondente
                })
        
        # 4. Remove duplicados (mantendo o primeiro encontrado)
        unique_matches = {}
        for info in matching_info:
            if info['index'] not in unique_matches:
                unique_matches[info['index']] = info['label']
        
        # 5. Converte para o formato de saída desejado
        result = {
            'indices': sorted(unique_matches.keys()),
            'labels': [unique_matches[idx] for idx in sorted(unique_matches.keys())]
        }
        
        return result

    def calculate_distances(self, X: np.array):
        # instructions to swap labels on Y
        indices_to_swap = {
            'indices': [],
            'labels': []
        }
        
        # Separating the curves of each class for better visualization
        curves_labeled = {class_label: X.get('curve') for class_label, X in self.X_separated.items()}
        
        # Getting the curve for each class
        for class_label, x in self.X_separated.items():
            # Create a temporary dataframe to store the distances
            distances_df = pd.DataFrame()
            
            # Get curve for each label
            for label, curve in curves_labeled.items():
                # Calculate the distances to the actual curve
                if (len(x['x_outliers']) != 0):
                    _, dists = curve.map_to_arcl(x['x_outliers'])
                    distances_df[label] = dists
                else:
                    distances_df[label] = []
        
            # Atribui o DataFrame completo de distâncias de volta ao X original
            adjusted_labels = distances_df.idxmin(axis=1)
            x_outlier_labeled = pd.DataFrame(x['x_outliers'])
            x_outlier_labeled['adjusted_labels'] = adjusted_labels
            
            # For each outlier assign the currect curve label
            aux = self.identify_indices_to_adjust(x_outlier_labeled, X)
            x['indices_to_swap'] = aux
            for key, content in aux.items():
                indices_to_swap[key].extend(content)
        
        self.indices_to_swap = indices_to_swap

    def change_labels_on_Y(self, indices_to_swap: dict, Y: np.array):
        # Ajusting indices
        for i, indice in enumerate(indices_to_swap['indices']):
                self.Y_adjusted[indice] = indices_to_swap['labels'][i]


# >>>>>>>>>>>>>>>>>>>>> TEST ONLY <<<<<<<<<<<<<<<<<<<<<<<<<<

def get_dataset_with_error(data, erro_proposto):

    def alterar_rotulos(Y, percentual, random_state=None):
        np.random.seed(random_state)  # Para reprodutibilidade
        Y_alterado = Y.copy()
        classes = np.unique(Y)

        for classe in classes:
            indices_classe = np.where(Y == classe)[0]  # Obtém índices da classe
            n_alterar = int(len(indices_classe) * percentual)
            indices_escolhidos = np.random.choice(
                indices_classe, n_alterar, replace=False
            )

            # Escolher novos rótulos aleatórios, diferentes do original
            for idx in indices_escolhidos:
                novas_classes = np.setdiff1d(classes, Y[idx])  # Evita o mesmo rótulo
                Y_alterado[idx] = np.random.choice(novas_classes)

        return Y_alterado

    X = data.data  # Features
    Y_original = data.target  # Rótulos

    # plot_completo(X, Y_original) # Plotando o gráfico original
    Y = alterar_rotulos(Y_original, erro_proposto)

    # plot_completo(X, Y) # Plotando o gráfico original

    # Remontar data_with_error
    data_with_error = {"data": data.data, "target": Y, "Y_original": Y_original}

    return data_with_error


if __name__ == "__main__":
    iris = load_iris()
    erro_proposto = 0.1
    results = {}
    iris_with_error = get_dataset_with_error(iris, erro_proposto)

    labels_erradas_antes_dos_ajustes = 0
    for i, y in enumerate(iris_with_error['target']):
        if y != iris.target[i]:
            labels_erradas_antes_dos_ajustes+=1

    lc = LabelCorrector()
    Y_adjusted = lc.run(X=iris_with_error["data"], Y=iris_with_error["target"])
    
    labels_erradas_depois_de_ajustadas = 0
    for i, y in enumerate(Y_adjusted):
        if y != iris.target[i]:
            labels_erradas_depois_de_ajustadas+=1
    
    a=1
