from sklearn.datasets import load_linnerud
from LabelCorrector.LabelCorrector import LabelCorrector
from utils.utils import get_dataset_with_error
import os
import numpy as np

if __name__ == "__main__":
    data = load_linnerud()
    erro_proposto = 0.1
    # Vamos trabalhar com apenas uma das saídas como classificação (ex: convertendo para classes)
    Y = data.target[:, 0]
    Y_classes = np.digitize(Y, bins=np.histogram_bin_edges(Y, bins=3))  # cria classes a partir dos valores contínuos

    data_with_error = get_dataset_with_error(data.data, Y_classes, erro_proposto)

    labels_wrong_before = sum(1 for i in range(len(Y_classes)) if data_with_error["target"][i] != Y_classes[i])
    print(f"Rótulos errados antes do ajuste: {labels_wrong_before}")

    lc = LabelCorrector()
    Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])

    lc.save_metrics_to_json_file(path='tests/linnerud/results_LabelCorrector_load_linnerud')

    for metric, value in lc.metrics.items():
        print(f"{metric}: {value}")
