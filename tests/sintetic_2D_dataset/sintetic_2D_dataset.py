import numpy as np
import matplotlib.pyplot as plt
from ocpc_py import MultiClassPC
import pandas as pd

from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file

def test_2D_sintetic_dataset():
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

    lc = PC_LabelCorrector()
    Y_adjusted = lc.run(X=np.array(data_with_error["data"]), Y=np.array(data_with_error["target"]).flatten())  # Flatten target

    # Comparação com o CL
    CL_issues = get_CL_label_correction(data_with_error["data"], data_with_error["target"], Y)

    # TODO: Gerar um arquivo json com as métricas do CL e do PC_labelCorrector para comparação
    # lc.save_metrics_to_json_file(path='tests/load_iris/results_LabelCorrector_load_iris')

    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues

    path='tests/sintetic_2D_dataset/comparation'
    # for metric, value in metrics.items():
#        print(f"{metric}: {value}")
        
    save_metrics_to_json_file(path=path, metrics=metrics)
    
    return metrics

if __name__=="__main__":
    test_2D_sintetic_dataset()