from ocpc_py import OneClassPC
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor


def alterar_rotulos(Y, percentual, random_state=None):
    np.random.seed(random_state)  # Para reprodutibilidade
    Y_alterado = Y.copy()
    classes = np.unique(Y)

    for classe in classes:
        indices_classe = np.where(Y == classe)[0]  # Obtém índices da classe
        n_alterar = int(len(indices_classe) * percentual)
        indices_escolhidos = np.random.choice(indices_classe, n_alterar, replace=False)

        # Escolher novos rótulos aleatórios, diferentes do original
        for idx in indices_escolhidos:
            novas_classes = np.setdiff1d(classes, Y[idx])  # Evita o mesmo rótulo
            Y_alterado[idx] = np.random.choice(novas_classes)

    return Y_alterado


def prepare_data_for_training(x_inlier):
    """Prepara os dados para o treinamento do modelo OneClassPC."""
    x_inlier_train, x_inlier_test = train_test_split(
        x_inlier, test_size=0.2, random_state=42
    )

    x_inlier_train = np.array(x_inlier_train)
    x_inlier_test = np.array(x_inlier_test)

    return x_inlier_train[:, 0:-1], x_inlier_test


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


df = sio.loadmat(
    "C:/Users/maxna/Documents/Projetos/Errors_in_labels_with_PC/utils/flame_data.mat"
)
df_x = df["X"]
df_y = df["y"]
X = np.concatenate((df_x, df_y), axis=1)
x_inlier = X[X[:, -1] == 1]
x_outlier = X[X[:, -1] == -1]

x_inlier_train, x_inlier_test = train_test_split(
    x_inlier, test_size=0.2, random_state=42
)


def main():
    # Adicionar erro proporsitalmente para teste
    iris = load_iris()
    X = iris.data  # Features
    Y_original = iris.target  # Rótulos
    Y = alterar_rotulos(Y_original, 0.3)
    # Remontar iris_with_error
    iris_with_error = {"data": iris.data, "target": Y}

    # Separar dataset em X e Y para cada classe diferente
    iris_data_separated_into_classes = {}
    for i in range(
        len(iris_with_error['data'])
    ):  # Percorrer todos os valores de X no dataset
        if iris.target[i] not in iris_data_separated_into_classes:
            iris_data_separated_into_classes[iris.target[i]] = {"X": []}
        iris_data_separated_into_classes[iris.target[i]]["X"].append(iris.data[i])

    # Aplicar o LOF em cada classe para encontrar os outliers
    for class_label, X in iris_data_separated_into_classes.items():
        X["predict"], X["_"] = detect_outliers_lof(
            X["X"], n_neighbors=int(len(X["X"]) / 2), contamination="auto"
        )
        # Gerar o x_clear e as outliers
        X["x_inliers"] = [x for i, x in enumerate(X["X"]) if X["predict"][i] == 1]
        X["x_outliers"] = [x for i, x in enumerate(X["X"]) if X["predict"][i] == -1]

        # TODO: Preparar os dados para treinamento do OneClass
        X["x_inlier_train"], X["x_inlier_test"] = prepare_data_for_training(
            x_inlier=X["x_inliers"]
        )
        # TODO: Rodar o OneClass com os inliers enconstrados pelo LOF para encontrar a CP
        X["curve"] = get_OneClass_curve(x_inlier_train=X["x_inlier_train"])

        # plotando a curva
        random_color = np.random.rand(
            3,
        )  # Gera três valores aleatórios entre 0 e 1 (RGB)
        fig, ax = plt.subplots()
        ax.scatter(
            np.array(X["X"])[:, 0],
            np.array(X["X"])[:, 1],
            color=random_color,
            label=class_label,
        )
        X["curve"].plot_curve(ax)

    # TODO: Para cada outlier calcular a distância até cada curva e atribuir o rótulo (label) da curva mais próxima


if __name__ == "__main__":
    main()
