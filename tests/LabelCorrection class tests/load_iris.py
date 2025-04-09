from sklearn.datasets import load_iris
from LabelCorrector.LabelCorrector import LabelCorrector
from utils.utils import get_dataset_with_error

if __name__ == "__main__":
    iris = load_iris()
    erro_proposto = 0.1
    results = {}
    iris_with_error = get_dataset_with_error(iris.data, iris.target, erro_proposto)

    labels_wrong_before_adjustments = 0
    for i, y in enumerate(iris_with_error['target']):
        if y != iris.target[i]:
            labels_wrong_before_adjustments += 1

    lc = LabelCorrector()
    Y_adjusted = lc.run(X=iris_with_error["data"], Y=iris_with_error["target"])
    
    lc.save_metrics_to_json_file(path='results_LabelCorrector_load_iris')
    
    for metric, value in lc.metrics.items():
        print(f"{metric}: {value}")