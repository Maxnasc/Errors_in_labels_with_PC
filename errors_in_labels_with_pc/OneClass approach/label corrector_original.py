import numpy as np
import pandas as pd
from ocpc_py import OneClassPC
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor


def separate_for_each_class(X: np.array, Y: np.array):
    # Dictionary of X values separated into classes
    x_separated = {}

    # Loop through the X values ​​by separating them
    for i in range(len(X)):
        if Y[i] not in x_separated:
            x_separated[Y[i]] = {"X": []}
        x_separated[Y[i]]["X"].append(np.array(X[i]))

    return x_separated

def separate_X_in_inliers_and_outliers(x_separated: dict):
    result = x_separated.copy()
    for class_label, X in x_separated.items():
        # Obtain outliers with LOF
        preditc, scores = detect_outliers_lof(
            X["X"], n_neighbors=int(len(X["X"]) / 2), contamination="auto"
        )
        # Identify inliers on X
        X["x_inliers"] = np.array(
            [x for i, x in enumerate(X["X"]) if preditc[i] == 1]
        )
        # Identify outliers on X
        X["x_outliers"] = np.array(
            [x for i, x in enumerate(X["X"]) if preditc[i] == -1]
        )
    return result


def detect_outliers_lof(X, n_neighbors=50, contamination="auto"):
    """Aplica o Local Outlier Factor (LOF) para detectar outliers."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_
    return y_pred, scores


def get_OneClass_curve(x_inlier_train):
    clf = OneClassPC()
    # removing the last column (target column)
    clf.fit(x_inlier_train)

    return clf.curve


def get_OneClassCurves(x_separated: dict):
    '''Obter as curvas para cada classe dentro de x_separated'''
    result = x_separated.copy()
    
    for class_label, X in result.items():
        X['curve'] = get_OneClass_curve(X.get('x_inliers')) # Obtain the curve for inliers data
    
    return result

def identify_indices_to_adjust(x_outlier_labeled, X):
    
    
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


def calculate_distances(x_separated: dict, X: np.array):
    # result variable for function modification
    result = x_separated.copy()
    # instructions to swap labels on Y
    indices_to_swap = {
        'indices': [],
        'labels': []
    }
    
    # Separating the curves of each class for better visualization
    curves_labeled = {class_label: X.get('curve') for class_label, X in x_separated.items()}
    
    # Getting the curve for each class
    for class_label, x in result.items():
        # Create a temporary dataframe to store the distances
        distances_df = pd.DataFrame()
        
        # Get curve for each label
        for label, curve in curves_labeled.items():
            # Calculate the distances to the actual curve
            _, dists = curve.map_to_arcl(x['x_outliers'])
            distances_df[label] = dists
    
        # Atribui o DataFrame completo de distâncias de volta ao X original
        adjusted_labels = distances_df.idxmin(axis=1)
        x_outlier_labeled = pd.DataFrame(x['x_outliers'])
        x_outlier_labeled['adjusted_labels'] = adjusted_labels
        
        # For each outlier assign the currect curve label
        aux = identify_indices_to_adjust(x_outlier_labeled, X)
        x['indices_to_swap'] = aux
        for key, content in aux.items():
            indices_to_swap[key].extend(content)
    
    return result, indices_to_swap


def change_labels_on_Y(indices_to_swap: dict, Y: np.array):
    y_adjusted = Y.copy()
    # Ajusting indices
    for i, indice in enumerate(indices_to_swap['indices']):
            y_adjusted[indice] = indices_to_swap['labels'][i]

    return y_adjusted

def run(X: np.array, Y: np.array):
    # Y adjusted
    Y_adjusted = np.array

    # Step 01: Separate X and Y according to each class
    X_separated = separate_for_each_class(X=X, Y=Y)

    # Step 02: Find the inliers and outliers
    X_separated = separate_X_in_inliers_and_outliers(x_separated=X_separated)
    
    # Step 03: Get the curves for each class with inliers and outliers
    X_separated = get_OneClassCurves(x_separated = X_separated)
    
    # Step 04 and 05: Calculate the distance from each outlier to each curve including the original curve
    X_separated, indices_to_swap = calculate_distances(x_separated=X_separated, X=X)
    
    # Step 06: Reassemble Y according to the adjustments made
    Y_adjusted = change_labels_on_Y(indices_to_swap=indices_to_swap, Y=Y)
    
    return Y_adjusted


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

    Y_adjusted = run(X=iris_with_error["data"], Y=iris_with_error["target"])
    
    labels_erradas_depois_de_ajustadas = 0
    for i, y in enumerate(Y_adjusted):
        if y != iris.target[i]:
            labels_erradas_depois_de_ajustadas+=1
    
    a=1
