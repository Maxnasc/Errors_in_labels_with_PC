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


def LabelCorrector(X: np.array, Y: np.array):

    # Step 01: Separate X and Y according to each class
    X_separated = separate_for_each_class(X=X, Y=Y)

    # Step 02: Find the inliers and outliers
    X_separated = separate_X_in_inliers_and_outliers(x_separated=X_separated)
    
    a = 1


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

    LabelCorrector(X=iris_with_error["data"], Y=iris_with_error["target"])
