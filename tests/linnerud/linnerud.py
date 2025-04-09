from sklearn.datasets import load_linnerud
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file
import os
import numpy as np

def test_linnerud_dataset():
    data = load_linnerud()
    erro_proposto = 0.1
    # Vamos trabalhar com apenas uma das saídas como classificação (ex: convertendo para classes)
    Y = data.target[:, 0]
    Y_classes = np.digitize(Y, bins=np.histogram_bin_edges(Y, bins=3))  # cria classes a partir dos valores contínuos

    data_with_error = get_dataset_with_error(data.data, Y_classes, erro_proposto)

    labels_wrong_before = sum(1 for i in range(len(Y_classes)) if data_with_error["target"][i] != Y_classes[i])
    print(f"Rótulos errados antes do ajuste: {labels_wrong_before}")

    lc = PC_LabelCorrector()
    Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])

    # lc.save_metrics_to_json_file(path='tests/linnerud/results_LabelCorrector_load_linnerud')

    # Comparação com o CL
    CL_issues = get_CL_label_correction(data_with_error["data"], data_with_error["target"], Y)
    
    # TODO: Gerar um arquivo json com as métricas do CL e do PC_labelCorrector para comparação
    # lc.save_metrics_to_json_file(path='tests/load_iris/results_LabelCorrector_load_iris')
    
    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues
    
    path='tests/load_iris/comparation'
    # for metric, value in metrics.items():
#        print(f"{metric}: {value}")
        
    save_metrics_to_json_file(path=path, metrics=metrics)
    
    return metrics

if __name__ == "__main__":
    test_linnerud_dataset()