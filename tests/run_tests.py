from tests.breast_cancer.breast_cancer import test_breast_cancer_dataset
from tests.digits.digits import test_digits_dataset
from tests.linnerud.linnerud import test_linnerud_dataset
from tests.load_iris.load_iris import test_load_iris_dataset
from tests.load_wine.load_wine import test_load_wine_dataset
from tests.sintetic_2D_dataset.sintetic_2D_dataset import test_2D_sintetic_dataset
from utils.utils import save_metrics_to_csv_file
import pandas as pd
import numpy as np
from codecarbon import EmissionsTracker
import os
import statistics

def get_statistics(global_metrics: dict, method_prefix: str):
    def get_indices(values, prefix):
        result = {
            f"{prefix}mean": np.mean([values]),
            f"{prefix}variance": np.var(values),  # por padrão, populacional
            f"{prefix}std": np.std(values)  # populacional
        }
        return result

    PC_LabelCorrection_rate = [dataset.get("original error rate PC_LabelCorrection", 0) for dataset_name, dataset in global_metrics.items()]
    CL_rate = [dataset.get("original error rate CL", 0) for dataset_name, dataset in global_metrics.items()]

    global_metrics[f'indices_PC_LabelCorrection_before_fix{method_prefix}'] = get_indices(PC_LabelCorrection_rate, prefix='before_fix_')
    global_metrics[f'indices_CL_before_fix{method_prefix}'] = get_indices(CL_rate, prefix='before_fix_')

    PC_LabelCorrection_rate = [dataset.get("error rate after correction PC_LabelCorrection", 0) for dataset_name, dataset in global_metrics.items()]
    CL_rate = [dataset.get("error rate after correction CL", 0) for dataset_name, dataset in global_metrics.items()]

    global_metrics[f'indices_PC_LabelCorrection_after_fix{method_prefix}'] = get_indices(PC_LabelCorrection_rate, prefix='after_fix_')
    global_metrics[f'indices_CL_after_fix{method_prefix}'] = get_indices(CL_rate, prefix='after_fix_')

    return global_metrics

def run_and_track_emissions(dataset_function, outlier_detection_ocpc: bool = None, method_name: str = "unknown"):
    tracker = EmissionsTracker(output_dir="./codecarbon_emissions", output_file=f"emissions_{method_name}.csv")
    tracker.start()
    if outlier_detection_ocpc is not None:
        metrics = dataset_function(outlier_detection_OCPC=outlier_detection_ocpc)
    else:
        metrics = dataset_function()
    tracker.stop()
    return metrics

def get_metrics_from_two_outlier_detection_method(n_samples: int):
    global_metrics_PC = {}
    global_metrics_LOF = {}

    # Executando os testes com PC_LabelCorrector (OCPC = True) e rastreando emissões
    print("Executando testes com PC_LabelCorrector...")
    try:
        global_metrics_PC['metric_2D'] = [run_and_track_emissions(test_2D_sintetic_dataset, outlier_detection_ocpc=True, method_name="PC_2D") for i in range(n_samples)]
    except Exception as e:
        global_metrics_PC['metric_2D'] = {"Erro": str(e)}

    try:
        global_metrics_PC['metric_breast_cancer'] = [run_and_track_emissions(test_breast_cancer_dataset, outlier_detection_ocpc=True, method_name="PC_breast_cancer") for i in range(n_samples)]
    except Exception as e:
        global_metrics_PC['metric_breast_cancer'] = {"Erro": str(e)}

    # try:
    #     global_metrics_PC['metric_digits'] = run_and_track_emissions(test_digits_dataset, outlier_detection_ocpc=True, method_name="PC_digits")
    # except Exception as e:
    #     global_metrics_PC['metric_digits'] = {"Erro": str(e)}

    # try:
    #     global_metrics_PC['metric_linnerud'] = run_and_track_emissions(test_linnerud_dataset, outlier_detection_ocpc=True, method_name="PC_linnerud")
    # except Exception as e:
    #     global_metrics_PC['metric_linnerud'] = {"Erro": str(e)}

    try:
        global_metrics_PC['metric_load_iris'] = [run_and_track_emissions(test_load_iris_dataset, outlier_detection_ocpc=True, method_name="PC_load_iris") for i in range(n_samples)]
    except Exception as e:
        global_metrics_PC['metric_load_iris'] = {"Erro": str(e)}

    try:
        global_metrics_PC['metric_load_wine'] = [run_and_track_emissions(test_load_wine_dataset, outlier_detection_ocpc=True, method_name="PC_load_wine") for i in range(n_samples)]
    except Exception as e:
        global_metrics_PC['metric_load_wine'] = {"Erro": str(e)}

    global_metrics_PC = get_statistics(global_metrics_PC, '_OCPC')

    #####################################################################

    # Executando os testes com LOF (OCPC = False) e rastreando emissões
    print("Executando testes com Confident Learning (LOF)...")
    try:
        global_metrics_LOF['metric_2D'] = [run_and_track_emissions(test_2D_sintetic_dataset, outlier_detection_ocpc=False, method_name="CL_2D") for i in range(n_samples)]
    except Exception as e:
        global_metrics_LOF['metric_2D'] = {"Erro": str(e)}

    try:
        global_metrics_LOF['metric_breast_cancer'] = [run_and_track_emissions(test_breast_cancer_dataset, outlier_detection_ocpc=False, method_name="CL_breast_cancer") for i in range(n_samples)]
    except Exception as e:
        global_metrics_LOF['metric_breast_cancer'] = {"Erro": str(e)}

    # try:
    #     global_metrics_LOF['metric_digits'] = run_and_track_emissions(test_digits_dataset, outlier_detection_ocpc=False, method_name="CL_digits")
    # except Exception as e:
    #     global_metrics_LOF['metric_digits'] = {"Erro": str(e)}

    # try:
    #     global_metrics_LOF['metric_linnerud'] = run_and_track_emissions(test_linnerud_dataset, outlier_detection_ocpc=False, method_name="CL_linnerud")
    # except Exception as e:
    #     global_metrics_LOF['metric_linnerud'] = {"Erro": str(e)}

    try:
        global_metrics_LOF['metric_load_iris'] = [run_and_track_emissions(test_load_iris_dataset, outlier_detection_ocpc=False, method_name="CL_load_iris") for i in range(n_samples)]
    except Exception as e:
        global_metrics_LOF['metric_load_iris'] = {"Erro": str(e)}

    try:
        global_metrics_LOF['metric_load_wine'] = [run_and_track_emissions(test_load_wine_dataset, outlier_detection_ocpc=False, method_name="CL_load_wine") for i in range(n_samples)]
    except Exception as e:
        global_metrics_LOF['metric_load_wine'] = {"Erro": str(e)}

    global_metrics_LOF = get_statistics(global_metrics_LOF, '_LOF')

    #####################################################################

    path = 'tests/global_metrics'

    global_metrics = {'PC': global_metrics_PC, 'LOF': global_metrics_LOF}
    
    def dict_to_csv_file(data: dict, nome_arquivo_csv: str):
        """
        Transforma um dicionário aninhado em um arquivo CSV legível.

        Args:
            data (dict): O dicionário aninhado a ser convertido.
            nome_arquivo_csv (str): O nome do arquivo CSV a ser criado.
        """
        linhas_csv = []
        for chave_externa, dicionario_interno in data.items():
            linha = {'Métrica': chave_externa}  # Primeira coluna com a chave externa
            if isinstance(dicionario_interno, dict):
                linha.update(dicionario_interno)  # Adiciona as chaves e valores do dicionário interno
            else:
                linha['Valor'] = dicionario_interno # Se o valor não for um dicionário, coloca em uma coluna 'Valor'
            linhas_csv.append(linha)

        dataframe = pd.DataFrame(linhas_csv)
        dataframe.to_csv(nome_arquivo_csv, index=False, encoding='utf-8')
        print(f"Arquivo CSV '{nome_arquivo_csv}' criado com sucesso.")    

    dict_to_csv_file(nome_arquivo_csv=f'{path}_PC.csv', data=global_metrics_PC)
    dict_to_csv_file(nome_arquivo_csv=f'{path}_LOF.csv', data=global_metrics_LOF)

    # Flatten the global_metrics dictionary
    flattened_data = []

    for detection_method, detection_metrics in global_metrics.items():
        for dataset_name, metrics in detection_metrics.items():
            if isinstance(metrics, dict):
                row = {'dataset': dataset_name, 'method': detection_method}
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            row[f"{key}_{sub_key}"] = sub_value
                    else:
                        row[key] = value
                flattened_data.append(row)

    # Cria DataFrame
    df = pd.DataFrame(flattened_data)

    # Salva como Excel
    df.to_excel('tests/global_metrics.xlsx', index=False)

if __name__=="__main__":
    get_metrics_from_two_outlier_detection_method(n_samples=1)