# 3. Aplicação do PCA

# Aplique PCA ao conjunto de dados normalizado e exiba a variância explicada por cada componente principal.

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = load_iris() # Carrega os dados do dataset de iris
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names) # Cria um dataframe com o dataset de iris

scaler = StandardScaler() # Scaler

iris_scaled = scaler.fit_transform(df_iris) # Normaliza os dados para serem usados dentro do PCA
df_iris_scaled = pd.DataFrame(iris_scaled, columns=iris.feature_names)

pca =  PCA(n_components=2) # Cria o objeto de pca com 2 componentes
pca.fit(iris_scaled) # Faz o PCA com a configuração correta

print(pca.explained_variance_ratio_	)
print(pca.explained_variance_)

iris_pca = pca.transform(iris_scaled)

plt.subplot(1, 2, 1)  # 1 linha, 2 colunas, 1º gráfico
plt.scatter(df_iris['sepal length (cm)'], df_iris['sepal width (cm)'], c=iris.target, cmap='viridis', edgecolor='k', alpha=0.7)
plt.title("Antes do PCA (Sepal Length vs Sepal Width)")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.colorbar(label="Classes")

plt.subplot(1, 2, 2)  # 1 linha, 2 colunas, 1º gráfico
plt.scatter(iris_pca[:,0], iris_pca[:,1], c=iris.target, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.xlabel('Componente 1 do PCA')
plt.ylabel('Componente 2 do PCA')
plt.title('Gráfico de dispersão do Iris Dataset')
plt.colorbar(label="Classes")

plt.tight_layout()
plt.show()