from sklearn.datasets import load_breast_cancer
from LabelCorrector.LabelCorrector import LabelCorrector
from utils.utils import get_dataset_with_error
import os

if __name__ == "__main__":
    data = load_breast_cancer()
    erro_proposto = 0.1
    data_with_error = get_dataset_with_error(data.data, data.target, erro_proposto)

    labels_wrong_before = sum(1 for i in range(len(data.target)) if data_with_error["target"][i] != data.target[i])
    print(f"Rótulos errados antes do ajuste: {labels_wrong_before}")

    lc = LabelCorrector()
    Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])

    lc.save_metrics_to_json_file(path='tests/breast_cancer/results_LabelCorrector_load_breast_cancer')

    for metric, value in lc.metrics.items():
        print(f"{metric}: {value}")
