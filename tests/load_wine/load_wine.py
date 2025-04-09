from sklearn.datasets import load_wine
from LabelCorrector.LabelCorrector import LabelCorrector
from utils.utils import get_dataset_with_error
import os

if __name__ == "__main__":
    wine = load_wine()
    erro_proposto = 0.1
    wine_with_error = get_dataset_with_error(wine.data, wine.target, erro_proposto)

    labels_wrong_before_adjustments = sum(
        1 for i in range(len(wine.target)) if wine_with_error["target"][i] != wine.target[i]
    )

    print(f"Rótulos errados antes do ajuste: {labels_wrong_before_adjustments}")

    lc = LabelCorrector()
    Y_adjusted = lc.run(X=wine_with_error["data"], Y=wine_with_error["target"])

    lc.save_metrics_to_json_file(path='tests/load_wine/results_LabelCorrector_load_wine')

    for metric, value in lc.metrics.items():
        print(f"{metric}: {value}")
