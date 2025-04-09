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
c1_out = np.zeros((c1.shape[0], 1))

xx = x + 2 
yy = -y + 6
c2 = np.concatenate((xx,yy), axis = 1)
c2_out = np.ones((c2.shape[0], 1))


plt.figure()
plt.plot(c1[:,0], c1[:,1], 'o')
plt.plot(c2[:,0], c2[:,1], 'r*')

X = np.concatenate((c1, c2), axis = 0)
Y = np.concatenate((c1_out, c2_out), axis = 0).flatten()  # Flatten Y to 1D

erro_proposto = 0.1
data_with_error = get_dataset_with_error(X, Y, erro_proposto)

lc = LabelCorrector()
Y_adjusted = lc.run(X=np.array(data_with_error["data"]), Y=np.array(data_with_error["target"]).flatten())  # Flatten target

lc.save_metrics_to_json_file(path='results_LabelCorrector_2D_dataset')

for metric, value in lc.metrics.items():
    print(f"{metric}: {value}")