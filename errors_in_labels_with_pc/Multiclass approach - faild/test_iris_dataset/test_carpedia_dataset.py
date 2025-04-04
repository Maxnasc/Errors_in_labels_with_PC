import numpy as np
import matplotlib.pyplot as plt 
from ocpc_py import MultiClassPC
import pandas as pd
import os

print(os.listdir())
df_dados_carpedia = pd.read_csv("errors_in_labels_with_pc/testing_2D_artificial_dataset/carpedia_error_label/dados.csv", sep=";")

# Extraindo dados do df
X = df_dados_carpedia.iloc[:, :-1]
# Alterando os valores em string de X para serem todos numéricos
X = X.map(lambda x: str(x).replace(',','.'))
X = X.map(pd.to_numeric)
X = X.values

Y = df_dados_carpedia.iloc[:,-1].values
Y = Y.ravel()

valores_classe_1 = [value for value in Y if value == 1]
valores_classe_menos_1 = [value for value in Y if value == -1]

clf = MultiClassPC(f=0.8)
clf.fit(X,Y)

curves_ = clf.curves

# Calcula as distâncias de cada ponto à cada curva
distances = {curve.label: curve.map_to_arcl(X) for curve in curves_}

# Junta as informações das distâncias para obter, em cada índice de dados do treinamento as distâncias para cada ponto
data_distances = pd.DataFrame({chave: dado[1] for chave, dado in distances.items()})
data_distances['original_label'] = Y
# Para cada ponto é verificada qual a classe mais próxima e comparada com a classe indicada no início, caso exista algum erro de rótulo o código altera a label do dado para melhor ajustar os dados
rows_to_adjust = []
for i, row in data_distances.iterrows():
    min_dist = row.iloc[:-1].idxmin(); # Classe cujo ponto tem menor distância
    min_value = row.iloc[:-1].min()
    if row['original_label'] != min_dist:
        rows_to_adjust.append(i)
        
# Ajustando as labels
for i in rows_to_adjust:
    if Y[i] == 1:
        Y[i] = -1 # Inverte 1 ↔ -1
    else:
        Y[i] = 1 # Inverte -1 ↔ 1

# Salva os dados ajustados em um novo DataFrame
data_distances['adjusted_label'] = Y
df_dados_ajustados = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
df_dados_ajustados['Label'] = Y

# Salvando no CSV
data_distances.to_excel('resultado_a_analizar.xlsx', index=False)
df_dados_ajustados.to_excel('dados_ajustados.xlsx', index=False)

print("Resultados salvos no arquivo 'resultado_a_analizar.xlsx'.")
print("Dados ajustados salvos no arquivo 'dados_ajustados.csv'.")
