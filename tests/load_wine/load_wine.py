import json
from sklearn.datasets import load_wine
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import calcula_novas_metricas, get_dataset_with_error, save_metrics_to_csv_file
import os
from codecarbon import EmissionsTracker

def run_label_correction(data, target, outlier_detection_ocpc: bool, tracker_prefix: str, k_max = int, alfa = float, lamda = float, f = float):
    # tracker = EmissionsTracker(output_dir="tests/load_wine/codecarbon_emissions", output_file=f"emissions_{tracker_prefix}.csv")
    # tracker.start()
    lc = PC_LabelCorrector(path='load_wine', detect_outlier_with_ocpc=outlier_detection_ocpc)
    Y_adjusted = lc.run(X=data, Y=target)
    # tracker.stop()
    return Y_adjusted, lc.metrics

def run_confident_learning(data, target, original_target, tracker_prefix: str):
    # tracker = EmissionsTracker(output_dir="tests/load_wine/codecarbon_emissions", output_file=f"emissions_{tracker_prefix}.csv")
    # tracker.start()
    cl_issues, issues = get_CL_label_correction(data, target, original_target)
    # tracker.stop()
    return cl_issues, issues

def test_load_wine_dataset(path: str, k_max = int, alfa = float, lamda = float, f = float, outlier_detection_OCPC=True):
    data = load_wine()
    erro_proposto = 0.1
    data_with_error = get_dataset_with_error(data.data, data.target, erro_proposto)

    labels_wrong_before = sum(1 for i in range(len(data.target)) if data_with_error["target"][i] != data.target[i])
    print(f"Rótulos errados antes do ajuste: {labels_wrong_before}")

    # Executando e rastreando emissões do PC_LabelCorrector
    Y_adjusted_pc, metrics_pc = run_label_correction(
        data_with_error["data"],
        data_with_error["target"],
        outlier_detection_OCPC,
        f"PC_2D_sintetic_{'OCPC' if outlier_detection_OCPC else 'LOF'}",
        k_max=k_max, alfa=alfa, lamda=lamda, f=f
    )

    # Executando e rastreando emissões do Confident Learning
    cl_issues, issues = run_confident_learning(
        data_with_error["data"],
        data_with_error["target"],
        data.get('target'),
        f"CL_2D_sintetic_{'OCPC' if outlier_detection_OCPC else 'LOF'}"
    )
    
    resultado_outliers_ocpc, resultado_outliers_CL = calcula_novas_metricas(path=path, outlier_detection_OCPC=outlier_detection_OCPC, Y=data.target, data_with_error=data_with_error, Y_adjusted_pc=Y_adjusted_pc, issues=issues)

    metrics = {
        "ocpc": resultado_outliers_ocpc,
        "CL": resultado_outliers_CL
    }
    # metrics = {"original error rate PC_LabelCorrection": metrics_pc['original error rate']} | {"error rate after correction PC_LabelCorrection": metrics_pc['error rate after correction']} | cl_issues

    path='tests/load_wine/comparation'
    save_metrics_to_csv_file(path=path, metrics=metrics)

    return metrics

if __name__ == "__main__":
    test_load_wine_dataset(path='load_wine', outlier_detection_OCPC=True)
    test_load_wine_dataset(path='load_wine', outlier_detection_OCPC=False)