from ocpc_py import OneClassPC
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"


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


def plot_completo(X, Y_original):
    # Criar um mapa de cores (ex: Set1 do Matplotlib)
    colors = plt.cm.Set1(np.linspace(0, 1, len(np.unique(Y_original))))

    # Criar o gráfico de dispersão para cada classe com uma cor diferente
    plt.figure(figsize=(8, 6))
    for i, class_label in enumerate(np.unique(Y_original)):
        plt.scatter(
            X[Y_original == class_label, 0],  # Primeiro atributo
            X[Y_original == class_label, 1],  # Segundo atributo
            color=colors[i],
            label=f"Classe {class_label}",
            alpha=0.7,  # Deixa os pontos levemente transparentes
            edgecolors="k",  # Adiciona borda preta aos pontos
        )

    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.title("Scatter Plot do Conjunto de Dados Iris")


def plot_curva(X, class_label):
    fig, ax = plt.subplots()
    ax.scatter(
        np.array(X["x_inliers"])[:, 0],
        np.array(X["x_inliers"])[:, 1],
        color="green",
        label=class_label,
    )
    if len(X["x_outliers"]) > 0:  # Só plota se houver outliers
        ax.scatter(
            np.array(X["x_outliers"])[:, 0],
            np.array(X["x_outliers"])[:, 1],
            color="red",
            label=f"{class_label} Outliers",
        )
    X["curve"].plot_curve(ax)


def calculate_distances(curve, x):
    # Converte lista de arrays em um único array NumPy 2D
    array_x = np.array(x)  

    # Chama a função com `x` ajustado
    distances = curve.map_to_arcl(array_x)  
    return distances

def main():
    # Adicionar erro proporsitalmente para teste
    iris = load_iris()
    X = iris.data  # Features
    Y_original = iris.target  # Rótulos

    # plot_completo(X, Y_original) # Plotando o gráfico original

    Y = alterar_rotulos(Y_original, 0.1)

    # plot_completo(X, Y) # Plotando o gráfico original

    # Remontar iris_with_error
    iris_with_error = {"data": iris.data, "target": Y}

    # Separar dataset em X e Y para cada classe diferente
    iris_data_separated_into_classes = {}
    for i in range(
        len(iris_with_error["data"])
    ):  # Percorrer todos os valores de X no dataset
        if iris_with_error["target"][i] not in iris_data_separated_into_classes:
            iris_data_separated_into_classes[iris_with_error["target"][i]] = {"X": []}
        iris_data_separated_into_classes[iris_with_error["target"][i]]["X"].append(
            iris_with_error["data"][i]
        )

    # Aplicar o LOF em cada classe para encontrar os outliers
    for class_label, X in iris_data_separated_into_classes.items():
        X["predict"], X["_"] = detect_outliers_lof(
            X["X"], n_neighbors=int(len(X["X"]) / 2), contamination="auto"
        )
        # Gerar o x_clear e as outliers
        X["x_inliers"] = [x for i, x in enumerate(X["X"]) if X["predict"][i] == 1]
        X["x_outliers"] = [x for i, x in enumerate(X["X"]) if X["predict"][i] == -1]
        X['outliers_percentage'] = len(X["x_outliers"])/(len(X["x_outliers"])+len(X["x_inliers"]))

        # Preparar os dados para treinamento do OneClass
        X["x_inlier_train"], X["x_inlier_test"] = prepare_data_for_training(
            x_inlier=X["x_inliers"]
        )
        # Rodar o OneClass com os inliers enconstrados pelo LOF para encontrar a CP
        X["curve"] = get_OneClass_curve(x_inlier_train=X["x_inlier_train"])
        
        calculate_distances(curve=X["curve"], x=X["x_inliers"])

        # plotando a curva
        plot_curva(X=X, class_label=class_label)

    # TODO: Para cada outlier calcular a distância até cada curva e atribuir o rótulo (label) da curva mais próxima
    
    average_percentage = sum([x['outliers_percentage'] for l, x in iris_data_separated_into_classes.items()])/len([x['outliers_percentage'] for l, x in iris_data_separated_into_classes.items()])
    print(round(average_percentage, 2))


if __name__ == "__main__":
    main()
    plt.show()
