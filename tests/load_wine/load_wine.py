from sklearn.datasets import load_wine
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file
import os

def test_load_wine_datset():
    wine = load_wine()
    erro_proposto = 0.1
    wine_with_error = get_dataset_with_error(wine.data, wine.target, erro_proposto)

    labels_wrong_before_adjustments = sum(
        1 for i in range(len(wine.target)) if wine_with_error["target"][i] != wine.target[i]
    )

    print(f"Rótulos errados antes do ajuste: {labels_wrong_before_adjustments}")

    lc = PC_LabelCorrector()
    Y_adjusted = lc.run(X=wine_with_error["data"], Y=wine_with_error["target"])

    # Comparação com o CL
    CL_issues = get_CL_label_correction(wine_with_error["data"], wine_with_error["target"], wine.target)
    
    # TODO: Gerar um arquivo json com as métricas do CL e do PC_labelCorrector para comparação
    # lc.save_metrics_to_json_file(path='tests/load_iris/results_LabelCorrector_load_iris')
    
    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues
    
    path='tests/load_wine/comparation'
    # for metric, value in metrics.items():
#        print(f"{metric}: {value}")
        
    save_metrics_to_json_file(path=path, metrics=metrics)
    
    return metrics

if __name__ == "__main__":
    test_load_wine_datset()