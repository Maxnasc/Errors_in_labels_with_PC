from sklearn.datasets import load_digits
from LabelCorrector.LabelCorrector import LabelCorrector
from utils.utils import get_dataset_with_error
import os

if __name__ == "__main__":
    data = load_digits()
    erro_proposto = 0.1
    data_with_error = get_dataset_with_error(data.data, data.target, erro_proposto)

    labels_wrong_before = sum(1 for i in range(len(data.target)) if data_with_error["target"][i] != data.target[i])
    print(f"Rótulos errados antes do ajuste: {labels_wrong_before}")

    lc = LabelCorrector()
    Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])

    lc.save_metrics_to_json_file(path='tests/digits/results_digits')

    for metric, value in lc.metrics.items():
        print(f"{metric}: {value}")

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