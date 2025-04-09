from sklearn.datasets import load_digits
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file
import os

def test_digits_dataset():
    data = load_digits()
    erro_proposto = 0.1
    data_with_error = get_dataset_with_error(data.data, data.target, erro_proposto)

    labels_wrong_before = sum(1 for i in range(len(data.target)) if data_with_error["target"][i] != data.target[i])
    print(f"Rótulos errados antes do ajuste: {labels_wrong_before}")

    lc = PC_LabelCorrector()
    Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])

    # Comparação com o CL
    CL_issues = get_CL_label_correction(data_with_error["data"], data_with_error["target"], data.target)
    
    # TODO: Gerar um arquivo json com as métricas do CL e do PC_labelCorrector para comparação
    # lc.save_metrics_to_json_file(path='tests/load_iris/results_LabelCorrector_load_iris')
    
    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues
    
    path='tests/digits/comparation'
    # for metric, value in metrics.items():
#        print(f"{metric}: {value}")
        
    save_metrics_to_json_file(path=path, metrics=metrics)
    
    return metrics

    # TODO: corrigir erro:
    # '''
    # Rótulos errados antes do ajuste: 176
    # Traceback (most recent call last):
    # File "c:\Users\maxna\Documents\Projetos\Errors_in_labels_with_PC\tests\LabelCorrection class tests\digits\digits.py", line 15, in <module>
    #     Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])
    #                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # File "C:\Users\maxna\Documents\Projetos\Errors_in_labels_with_PC\LabelCorrector\LabelCorrector.py", line 295, in run
    #     self.X_separated, self.indexes_to_swap = self._calculate_distances(
    #                                             ^^^^^^^^^^^^^^^^^^^^^^^^^^
    # File "C:\Users\maxna\Documents\Projetos\Errors_in_labels_with_PC\LabelCorrector\LabelCorrector.py", line 168, in _calculate_distances
    #     _, dists = curve.map_to_arcl(x["x_outliers"])
    #             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # File "C:\Users\maxna\Documents\Projetos\Errors_in_labels_with_PC\.venv\Lib\site-packages\ocpc_py\Models.py", line 241, in map_to_arcl
    # '''
    
if __name__ == "__main__":
    test_digits_dataset()