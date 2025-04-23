from sklearn.datasets import load_breast_cancer
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file
import os

def test_breast_cancer_dataset(outlier_detection_OCPC: bool):
    data = load_breast_cancer()
    erro_proposto = 0.1
    data_with_error = get_dataset_with_error(data.data, data.target, erro_proposto)

    labels_wrong_before = sum(1 for i in range(len(data.target)) if data_with_error["target"][i] != data.target[i])
    print(f"Rótulos errados antes do ajuste: {labels_wrong_before}")

    lc = PC_LabelCorrector(detect_outlier_with_ocpc=outlier_detection_OCPC)
    Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])

    # Comparação com o CL
    CL_issues = get_CL_label_correction(data_with_error["data"], data_with_error["target"], data.target)
    
    # TODO: Gerar um arquivo json com as métricas do CL e do PC_labelCorrector para comparação
    # lc.save_metrics_to_json_file(path='tests/load_iris/results_LabelCorrector_load_iris')
    
    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues
    
    path='tests/breast_cancer/comparation'
    # for metric, value in metrics.items():
#        print(f"{metric}: {value}")
        
    save_metrics_to_json_file(path=path, metrics=metrics)
    
    return metrics
    
if __name__ == "__main__":
    test_breast_cancer_dataset()