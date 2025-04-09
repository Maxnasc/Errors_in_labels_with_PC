from sklearn.datasets import load_iris
from LabelCorrector.LabelCorrector import LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file

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
    
    # Comparação com o CL
    CL_issues = get_CL_label_correction(iris_with_error["data"], iris_with_error["target"], iris.target)
    
    # TODO: Gerar um arquivo json com as métricas do CL e do PC_labelCorrector para comparação
    # lc.save_metrics_to_json_file(path='tests/load_iris/results_LabelCorrector_load_iris')
    
    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues
    
    path='tests/load_iris/comparation'
    for metric, value in metrics.items():
        print(f"{metric}: {value}")
        
    save_metrics_to_json_file(path=path, metrics=metrics)