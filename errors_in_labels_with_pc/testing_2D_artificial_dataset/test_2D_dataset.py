import numpy as np
import matplotlib.pyplot as plt 
from ocpc_py import MultiClassPC
import pandas as pd

x = np.linspace(-2, 2, num=101)
media_ruido = 0; 
var_ruido = 0.5
ruido = media_ruido + (var_ruido * np.random.randn(x.shape[0]))
y = x**2 + ruido

x = x[:,np.newaxis]; y = y[:,np.newaxis]
c1 = np.concatenate((x,y), axis = 1)
c1_out =  np.zeros((c1.shape[0], 1))

xx = x + 2 
yy = -y + 6
c2 = np.concatenate((xx,yy), axis = 1)
c2_out = np.ones((c2.shape[0], 1))


plt.figure()
plt.plot(c1[:,0], c1[:,1], 'o')
plt.plot(c2[:,0], c2[:,1], 'r*')

X = np.concatenate((c1, c2), axis = 0)
Y = np.concatenate((c1_out, c2_out), axis = 0)

clf = MultiClassPC(f=0.8)
clf.fit(X,Y.flatten())

curves_ = clf.curves

# Calcula as distâncias de cada ponto à cada curva
distances = {i: curve.map_to_arcl(X) for i, curve in enumerate(curves_)}

# Junta as informações das distâncias para obter, em cada índice de dados do treinamento as distâncias para cada ponto
data_distances = pd.DataFrame({chave: dado[1] for chave, dado in distances.items()})
data_distances['original_labels'] = Y
# Para cada ponto é verificada qual a classe mais próxima e comparada com a classe indicada no início, caso exista algum erro de rótulo o código altera a label do dado para melhor ajustar os dados
# for i, row in data_distances.iterrows():
    

a=1

fig, ax = plt.subplots()
ax.plot(c1[:,0], c1[:,1], 'o')
ax.plot(c2[:,0], c2[:,1], 'r*')
curves_[0].plot_curve(ax)
curves_[1].plot_curve(ax)

# print(distances)
plt.show()