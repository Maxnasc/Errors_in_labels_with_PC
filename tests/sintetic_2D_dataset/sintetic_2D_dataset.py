import numpy as np
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_csv_file
import os
from codecarbon import EmissionsTracker

def run_label_correction(data, target, outlier_detection_ocpc: bool, tracker_prefix: str):
    tracker = EmissionsTracker(output_dir="tests/sintetic_2D_dataset/codecarbon_emissions", output_file=f"emissions_{tracker_prefix}.csv")
    tracker.start()
    lc = PC_LabelCorrector(detect_outlier_with_ocpc=outlier_detection_ocpc)
    Y_adjusted = lc.run(X=data, Y=target)
    tracker.stop()
    return Y_adjusted, lc.metrics

def run_confident_learning(data, target, original_target, tracker_prefix: str):
    tracker = EmissionsTracker(output_dir="tests/sintetic_2D_dataset/codecarbon_emissions", output_file=f"emissions_{tracker_prefix}.csv")
    tracker.start()
    cl_issues = get_CL_label_correction(data, target, original_target)
    tracker.stop()
    return cl_issues

def test_2D_sintetic_dataset(outlier_detection_OCPC: bool):
    x = np.linspace(-2, 2, num=101)
    media_ruido = 0; 
    var_ruido = 0.8
    ruido = media_ruido + (var_ruido * np.random.randn(x.shape[0]))
    y = x**2 # Ruido removido

    x = x[:,np.newaxis]; y = y[:,np.newaxis]
    c1 = np.concatenate((x,y), axis = 1)
    c1_out = np.zeros((c1.shape[0], 1))

    xx = x + 2 
    yy = -y + 6
    c2 = np.concatenate((xx,yy), axis = 1)
    c2_out = np.ones((c2.shape[0], 1))

    X = np.concatenate((c1, c2), axis = 0)
    Y = np.concatenate((c1_out, c2_out), axis = 0).flatten()  # Flatten Y to 1D

    erro_proposto = 0.1
    data_with_error = get_dataset_with_error(X, Y, erro_proposto)

    labels_wrong_before = sum(1 for i in range(len(Y)) if data_with_error["target"][i] != Y[i])
    print(f"Rótulos errados antes do ajuste: {labels_wrong_before}")

    # Executando e rastreando emissões do PC_LabelCorrector
    Y_adjusted_pc, metrics_pc = run_label_correction(
        data_with_error["data"],
        data_with_error["target"],
        outlier_detection_OCPC,
        f"PC_2D_sintetic_{'OCPC' if outlier_detection_OCPC else 'LOF'}"
    )

    # Executando e rastreando emissões do Confident Learning
    cl_issues = run_confident_learning(
        data_with_error["data"],
        data_with_error["target"],
        Y,
        f"CL_2D_sintetic_{'OCPC' if outlier_detection_OCPC else 'LOF'}"
    )

    metrics = {"original error rate PC_LabelCorrection": metrics_pc['original error rate']} | {"error rate after correction PC_LabelCorrection": metrics_pc['error rate after correction']} | cl_issues

    path='tests/sintetic_2D_dataset/comparation'
    save_metrics_to_csv_file(path=path, metrics=metrics)

    return metrics

if __name__ == "__main__":
    test_2D_sintetic_datset(outlier_detection_OCPC=True)
    test_2D_sintetic_datset(outlier_detection_OCPC=False)