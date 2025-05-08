from tests.breast_cancer.breast_cancer import test_breast_cancer_dataset
from tests.digits.digits import test_digits_dataset
from tests.linnerud.linnerud import test_linnerud_dataset
from tests.load_iris.load_iris import test_load_iris_datset
from tests.load_wine.load_wine import test_load_wine_datset
from tests.sintetic_2D_dataset.sintetic_2D_dataset import test_2D_sintetic_dataset
from utils.utils import save_metrics_to_json_file
import pandas as pd
import numpy as np

def get_statistics(global_metrics: dict, method_prefix: str):
    def get_indices(values, prefix):
        result = {
            f"{prefix}mean": np.mean(values),
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

def get_metrics_from_two_outlier_detection_method():
    # Run all the dataset tests, merge metrics results and grab diferences (global metrics)
    global_metrics_PC = {}

    try:
        global_metrics_PC['metric_2D'] = test_2D_sintetic_dataset(outlier_detection_OCPC=True)
    except Exception:
        global_metrics_PC['metric_2D'] = {"Erro": str(Exception)}
        
    try:
        global_metrics_PC['metric_breast_cancer'] = test_breast_cancer_dataset(outlier_detection_OCPC=True)
    except Exception:
        global_metrics_PC['metric_breast_cancer'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics_PC['metric_digits'] = test_digits_dataset(outlier_detection_OCPC=True)
    # except Exception:
    #     global_metrics_PC['metric_digits'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics_PC['metric_linnerud'] = test_linnerud_dataset(outlier_detection_OCPC=True)
    # except Exception:
    #     global_metrics_PC['metric_linnerud'] = {"Erro": str(Exception)}
        
    try:
        global_metrics_PC['metric_load_iris'] = test_load_iris_datset(outlier_detection_OCPC=True)
    except Exception:
        global_metrics_PC['metric_load_iris'] = {"Erro": str(Exception)}
        
    try:
        global_metrics_PC['metric_load_wine'] = test_load_wine_datset(outlier_detection_OCPC=True)
    except Exception:
        global_metrics_PC['metric_load_wine'] = {"Erro": str(Exception)}
        
    global_metrics_PC = get_statistics(global_metrics_PC, '_OCPC')
    
    #####################################################################
    
    global_metrics_LOF = {}

    try:
        global_metrics_LOF['metric_2D'] = test_2D_sintetic_dataset(outlier_detection_OCPC=False)
    except Exception:
        global_metrics_LOF['metric_2D'] = {"Erro": str(Exception)}
        
    try:
        global_metrics_LOF['metric_breast_cancer'] = test_breast_cancer_dataset(outlier_detection_OCPC=False)
    except Exception:
        global_metrics_LOF['metric_breast_cancer'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics_LOF['metric_digits'] = test_digits_dataset()
    # except Exception:
    #     global_metrics_LOF['metric_digits'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics_LOF['metric_linnerud'] = test_linnerud_dataset()
    # except Exception:
    #     global_metrics_LOF['metric_linnerud'] = {"Erro": str(Exception)}
        
    try:
        global_metrics_LOF['metric_load_iris'] = test_load_iris_datset(outlier_detection_OCPC=False)
    except Exception:
        global_metrics_LOF['metric_load_iris'] = {"Erro": str(Exception)}
        
    try:
        global_metrics_LOF['metric_load_wine'] = test_load_wine_datset(outlier_detection_OCPC=False)
    except Exception:
        global_metrics_LOF['metric_load_wine'] = {"Erro": str(Exception)}
        
    global_metrics_LOF = get_statistics(global_metrics_LOF, '_LOF')
    
    
    #####################################################################
    
    path = 'tests/global_metrics'
    
    global_metrics = {'PC': global_metrics_PC, 'LOF': global_metrics_LOF}
    
    save_metrics_to_json_file(path=path, metrics=global_metrics)
    # print(global_metrics)

    # Flatten the global_metrics dictionary
    flattened_data = []

    for detection_method, detection_metrics in global_metrics.items():
        for dataset_name, metrics in detection_metrics.items():
            if isinstance(metrics, dict):
                row = {'dataset': dataset_name}
                for key, value in metrics.items():
                    if isinstance(value, dict):
                        # indices_PC_LabelCorrection, indices_CL, etc
                        for sub_key, sub_value in value.items():
                            row[f"{key}_{sub_key}"] = sub_value
                    else:
                        row[key] = value
                flattened_data.append(row)

    # Cria DataFrame
    df = pd.DataFrame(flattened_data)

    # Salva como CSV
    # df.to_csv('tests/global_metrics/global_metrics.csv', index=False)

    # Salva como Excel
    df.to_excel('tests/global_metrics.xlsx', index=False)




if __name__=="__main__":
    # # Run all the dataset tests, merge metrics results and grab diferences (global metrics)
    # global_metrics = {}

    # try:
    #     global_metrics['metric_2D'] = test_2D_sintetic_dataset()
    # except Exception:
    #     global_metrics['metric_2D'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics['metric_breast_cancer'] = test_breast_cancer_dataset()
    # except Exception:
    #     global_metrics['metric_breast_cancer'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics['metric_digits'] = test_digits_dataset()
    # except Exception:
    #     global_metrics['metric_digits'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics['metric_linnerud'] = test_linnerud_dataset()
    # except Exception:
    #     global_metrics['metric_linnerud'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics['metric_load_iris'] = test_load_iris_datset()
    # except Exception:
    #     global_metrics['metric_load_iris'] = {"Erro": str(Exception)}
        
    # try:
    #     global_metrics['metric_load_wine'] = test_load_wine_datset()
    # except Exception:
    #     global_metrics['metric_load_wine'] = {"Erro": str(Exception)}
        
    # global_metrics = get_statistics(global_metrics)
    
    # path = 'tests/global_metrics'
    # save_metrics_to_json_file(path=path, metrics=global_metrics)
    # # print(global_metrics)

    # # Flatten the global_metrics dictionary
    # flattened_data = []

    # for dataset_name, metrics in global_metrics.items():
    #     if isinstance(metrics, dict):
    #         row = {'dataset': dataset_name}
    #         for key, value in metrics.items():
    #             if isinstance(value, dict):
    #                 # indices_PC_LabelCorrection, indices_CL, etc
    #                 for sub_key, sub_value in value.items():
    #                     row[f"{key}_{sub_key}"] = sub_value
    #             else:
    #                 row[key] = value
    #         flattened_data.append(row)

    # # Cria DataFrame
    # df = pd.DataFrame(flattened_data)

    # # Salva como CSV
    # # df.to_csv('tests/global_metrics/global_metrics.csv', index=False)

    # # Salva como Excel
    # df.to_excel('tests/global_metrics.xlsx', index=False)
    
    get_metrics_from_two_outlier_detection_method()
