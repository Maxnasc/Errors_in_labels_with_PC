from ocpc_py import OneClassPC
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.neighbors import LocalOutlierFactor
import os
import tqdm

from utils.confident_learning import get_CL_label_correction
import json

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

    # x_inlier_train = np.array(x_inlier_train)
    # x_inlier_test = np.array(x_inlier_test)

    return x_inlier_train, x_inlier_test


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
        X["x_inliers"][:, 0],
        X["x_inliers"][:, 1],
        color="green",
        label=class_label,
    )
    if len(X["x_outliers"]) > 0:  # Só plota se houver outliers
        ax.scatter(
            X["x_outliers"][:, 0],
            X["x_outliers"][:, 1],
            color="red",
            label=f"{class_label} Outliers",
        )
    X["curve"].plot_curve(ax)


def calculate_distances(curve, x):
    # Converte lista de arrays em um único array NumPy 2D
    # array_x = np.array(x)  

    # Chama a função com `x` ajustado
    distances = curve.map_to_arcl(x)  
    return distances

def identify_indices_to_adjust(x_adjusted, X):
    # 1. Extrai os valores numéricos e os labels
    x_adjusted_values = x_adjusted.iloc[:, :-1].values
    x_adjusted_labels = x_adjusted.iloc[:, -1].values  # Pega a coluna adjusted_labels
    
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


def get_dataset_with_error(data, erro_proposto):
    X = data.data  # Features
    Y_original = data.target  # Rótulos

    # plot_completo(X, Y_original) # Plotando o gráfico original
    Y = alterar_rotulos(Y_original, erro_proposto)

    # plot_completo(X, Y) # Plotando o gráfico original

    # Remontar data_with_error
    data_with_error = {"data": data.data, "target": Y, "Y_original": Y_original}
    
    return data_with_error

def main(data_with_error):
    Y = data_with_error['target']
    Y_original = data_with_error['Y_original']
    
    # Informações do resultado
    labels_erradas = 0
    melhora_CP = 0
    result_CL = 0
    melhora_CL = 0
    
    # Separar dataset em X e Y para cada classe diferente
    iris_data_separated_into_classes = {}
    for i in range(
        len(data_with_error["data"])
    ):  # Percorrer todos os valores de X no dataset
        if data_with_error["target"][i] not in iris_data_separated_into_classes:
            iris_data_separated_into_classes[data_with_error["target"][i]] = {"X": []}
        iris_data_separated_into_classes[data_with_error["target"][i]]["X"].append(
            np.array(data_with_error["data"][i])
        )

    # Aplicar o LOF em cada classe para encontrar os outliers
    for class_label, X in iris_data_separated_into_classes.items():
        X["predict"], X["_"] = detect_outliers_lof(
            X["X"], n_neighbors=int(len(X["X"]) / 2), contamination="auto"
        )
        # Gerar o x_clear e as outliers
        X["x_inliers"] = np.array([x for i, x in enumerate(X["X"]) if X["predict"][i] == 1])
        X["x_outliers"] = np.array([x for i, x in enumerate(X["X"]) if X["predict"][i] == -1])
        X['outliers_percentage'] = len(X["x_outliers"])/(len(X["x_outliers"])+len(X["x_inliers"]))

        # Preparar os dados para treinamento do OneClass
        X["x_inlier_train"], X["x_inlier_test"] = prepare_data_for_training(
            x_inlier=X["x_inliers"]
        )
        # Rodar o OneClass com os inliers enconstrados pelo LOF para encontrar a CP
        X["curve"] = get_OneClass_curve(x_inlier_train=X["x_inlier_train"])
        
        calculate_distances(curve=X["curve"], x=X["x_outliers"])

        # plotando a curva
        plot_curva(X=X, class_label=class_label)

    # Separando as curvas de cada classe para melhor visualização
    curves_labeled = {class_label: X.get('curve') for class_label, X in iris_data_separated_into_classes.items()}
    
    # TODO: Para cada outlier calcular a distância até cada curva e atribuir o rótulo (label) da curva mais próxima
    Y_adjusted = Y.copy()
    for class_label, X in iris_data_separated_into_classes.items():
        # Cria um DataFrame temporário para as distâncias
        # distances_df = pd.DataFrame(X['x_outliers'][:, 0:-1])
        distances_df = pd.DataFrame()
        
        for label, curve in curves_labeled.items():
            # Calcula as distâncias para a curva atual
            _, dists = curve.map_to_arcl(X['x_outliers'])  # Assumindo que calculate_distances chama map_to_arcl
            distances_df[label] = dists  # Adiciona como nova coluna
    
        # Atribui o DataFrame completo de distâncias de volta ao X original
        X['x_outlier_distances'] = distances_df
        adjusted_labels = distances_df.idxmin(axis=1)
        X['x_outlier_labeled'] = pd.DataFrame(X['x_outliers'])
        X['x_outlier_labeled']['adjusted_labels']= adjusted_labels
        # Para cada outlier eu vou identificar a classe mais próxima e atribuir a label referente
        indices = identify_indices_to_adjust(X['x_outlier_labeled'], data_with_error['data'])
        
        for i, indice in enumerate(indices['indices']):
            Y_adjusted[indice] = indices['labels'][i]
    
    labels_erradas = 0
    for i, y in enumerate(Y_adjusted):
        if y != Y_original[i]:
            labels_erradas+=1
    
        
    error_percentage = round((labels_erradas/len(Y_original)*100), 2)
    erro_proposto_formatado = round((erro_proposto*100), 2)
    melhora_CP = round(((1-(labels_erradas/len(Y_original))/erro_proposto)*100), 2)
    
    result_CL = get_CL_label_correction(X=iris.data, Y_error=Y, Y_original=Y_original)
    error_percentage_CL = round((result_CL/len(Y_original)*100), 2)
    melhora_CL = round(((1-(result_CL/len(Y_original))/erro_proposto)*100), 2)
    
    # print()
    # print(f"Erro proposto: {erro_proposto_formatado}%")
    
    # print("Métricas do ajustador de rótulos com curvas principais:")
    # print(f"Erro depois de processado: {error_percentage}%")
    # print(f"Melhora de {melhora_CP}% nos erros")
    
    # print("Métricas do ajustador de rótulos com confident learning:")
    # print(f"Erro depois de processado: {error_percentage_CL}%")
    # print(f"Melhora de {melhora_CL}% nos erros")
    
    return {
        "labels_erradas_CP": labels_erradas,
        "melhora_CP": melhora_CP,
        "labels_erradas_CL": result_CL,
        "melhora_CL": melhora_CL,
    }


if __name__ == "__main__":
    iris = load_iris()
    erro_proposto = 0.1
    repeticoes = 50
    results = {}
    for i in tqdm.tqdm(range(repeticoes), desc="Fixing labels"):
        iris_with_error = get_dataset_with_error(iris, erro_proposto)
        results[i]=(main(iris_with_error))
    
    for idx, result in results.items():
        for key, value in result.items():
            result[key] = float(value)
    
    metrics = {}
    keys = results[0].keys()
    for key in keys:
        values = [result[key] for idx, result in results.items()]
        metrics[key] = {
            "mean": float(np.mean(values)),  # Convert to float
            "variance": float(np.var(values)),  # Convert to float
            "std_dev": float(np.std(values)),  # Convert to float
        }
        
    results['metrics'] = metrics

    
    # Save results to a JSON file
    output_file = "50_avaliacoes_do_load_iris_com_diferentes_configurações_de_erros_10%_de_erro.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_file}")
    plt.show()
