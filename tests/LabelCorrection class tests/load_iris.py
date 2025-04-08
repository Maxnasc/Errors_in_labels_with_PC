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
    
    labels_wrong_after_adjustments = 0
    for i, y in enumerate(Y_adjusted):
        if y != iris.target[i]:
            labels_wrong_after_adjustments += 1
    
    print(f"labels_wrong_before_adjustments {labels_wrong_before_adjustments}")
    print(f"labels_wrong_after_adjustments {labels_wrong_after_adjustments}")
    print(f"Erro diminuido de {erro_proposto*100}% para {round((labels_wrong_after_adjustments/len(iris_with_error['data'])), 3)*100}%")