import numpy as np
import matplotlib.pyplot as plt
from ocpc_py import MultiClassPC
import pandas as pd

from LabelCorrector.LabelCorrector import LabelCorrector
from utils.utils import get_dataset_with_error

x = np.linspace(-2, 2, num=101)
media_ruido = 0; 
var_ruido = 0.8
ruido = media_ruido + (var_ruido * np.random.randn(x.shape[0]))
y = x**2 # Ruido removido

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

erro_proposto = 0.1
data_with_error = get_dataset_with_error(X, Y, erro_proposto)

labels_wrong_before_adjustments = 0
for i, y in enumerate(data_with_error['target']):
    if y != Y[i]:
        labels_wrong_before_adjustments += 1

lc = LabelCorrector()
Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])

labels_wrong_after_adjustments = 0
for i, y in enumerate(Y_adjusted):
    if y != Y[i]:
        labels_wrong_after_adjustments += 1

print(f"labels_wrong_before_adjustments {labels_wrong_before_adjustments}")
print(f"labels_wrong_after_adjustments {labels_wrong_after_adjustments}")
print(f"Erro diminuido de {erro_proposto*100}% para {round((labels_wrong_after_adjustments/len(iris_with_error['data'])), 3)*100}%")