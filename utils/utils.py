import json
import numpy as np
import pandas as pd
from copy import deepcopy

def get_dataset_with_error(X, Y_original, erro_proposto):

    def alterar_rotulos(Y, percentual, random_state=None):
        """
        Alters the labels of Y by a given percentage.

        Args:
            Y: Original labels
            percentual: Percentage of labels to alter
            random_state: Seed for reproducibility

        Returns:
            Altered labels
        """
        np.random.seed(random_state)  # For reproducibility
        Y_altered = Y.copy()
        classes = np.unique(Y)

        for classe in classes:
            class_indices = np.where(Y == classe)[0]  # Get indices of the class
            n_to_alter = int(len(class_indices) * percentual)
            chosen_indices = np.random.choice(
                class_indices, n_to_alter, replace=False
            )

            # Choose new random labels, different from the original
            for idx in chosen_indices:
                new_classes = np.setdiff1d(classes, Y[idx])  # Avoid the same label
                Y_altered[idx] = np.random.choice(new_classes)

        return Y_altered

    Y = alterar_rotulos(Y_original, erro_proposto)

    # Rebuild data_with_error
    data_with_error = {"data": X, "target": Y, "Y_original": Y_original}

    return data_with_error

def save_metrics_to_json_file(path: str, metrics: dict):
    
        # Save results to a JSON file
        if ".json" not in path:
            path = path + ".json"

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Results saved to {path}")
        
def save_metrics_to_csv_file(path: str, metrics: dict):

        # Coverter os valores dentro de metrics para listas
        new_metrics = deepcopy(metrics)
        for key, value in new_metrics.items():
            list_val = [value]
            new_metrics[key] = list_val
            
        df = pd.DataFrame(new_metrics)
        
        if ".csv" not in path:
            path = path + ".csv"
        
        df.to_csv(path);

        print(f"Results saved to {path}")
        
def calcula_novas_metricas(path, outlier_detection_OCPC, Y, data_with_error, Y_adjusted_pc, issues):
    # entender quais labels estão errados
    erros = [i for i, value in enumerate(Y) if value != data_with_error.get('target')[i]]
    
    if outlier_detection_OCPC:
        caminho_ouliers_pc = f'tests/{path}/outliers_ocpc.json'
        caminho_ocpc = f'tests/{path}/outliers_result_ocpc.json'
        caminho_cl = f'tests/{path}/outliers_result_cl.json'
    else:
        caminho_ouliers_pc = f'tests/{path}/outliers_lof.json'
        caminho_ocpc = f'tests/{path}/outliers_result_lof.json'
        caminho_cl = f'tests/{path}/outliers_result_cl.json'
    
    with open(caminho_ouliers_pc, 'r') as opcl:
        outliers_saved = json.load(opcl)    
        
    correct_detected_outliers = [o for i, o in enumerate(outliers_saved) if o == -1 and i in erros]
    wrong_detected_outliers = [o for i, o in enumerate(outliers_saved) if o == -1 and i not in erros]
    
    # Qual a porcentagem do erro que foi efetivamente corrigida ?
    erros_depois_de_corrigir = [i for i, value in enumerate(Y) if value != Y_adjusted_pc[i]]
    
    erros_ajustados_corretamente = [i for i in erros if i not in erros_depois_de_corrigir]
    erros_nao_corrigidos = [i for i in erros if i in erros_depois_de_corrigir]
    novos_erros_gerados = [i for i in erros_depois_de_corrigir if i not in erros]  

    # Cálculos para resultado_outliers_CL, espelhando a lógica do OCPC
    # issues['erro_original'] = issues['original_labels'] != data_with_error.get('target')
    # issues['erro_apos_correcao'] = issues['original_labels'] != issues['predicted_label']

    # erros_indices_CL = issues[issues['erro_original']].index.tolist()
    # erros_depois_corrigir_indices_CL = issues[issues['erro_apos_correcao']].index.tolist()

    # erros_ajustados_corretamente_CL = [i for i in erros_indices_CL if i not in erros_depois_corrigir_indices_CL]
    # erros_nao_corrigidos_CL = [i for i in erros_indices_CL if i in erros_depois_corrigir_indices_CL]
    # novos_erros_gerados_CL = [i for i in erros_depois_corrigir_indices_CL if i not in erros_indices_CL]

    # correct_outliers_detected_CL = issues[issues['is_label_issue'] & (issues['given_label'] == issues['original_labels'])]
    # wrong_false_alarm_CL = issues[issues['is_label_issue'] & (issues['given_label'] != issues['original_labels'])]
    
    resultado_outliers_ocpc = {
        'taxa_de_erro_detectada_corretamente': len(correct_detected_outliers)/len(Y),
        'taxa_de_erro_detectada_erradamente': len(wrong_detected_outliers)/len(Y),
        'erros_de_rotulo_ajustados_corretamente': len(erros_ajustados_corretamente),
        'taxa_do_erro_ajustada_corretamente': len(erros_ajustados_corretamente)/len(erros),
        'taxa_do_erro_nao_corrigida': len(erros_nao_corrigidos)/len(erros),
        'novos_erros_gerados': len(novos_erros_gerados),
        'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': len(novos_erros_gerados)/len(Y),
    }
    
    resultado_outliers_CL = {
        'taxa_de_erro_detectada_corretamente': 'correct_outliers_detected_CL.shape[0] / len(Y)',
        'taxa_de_erro_detectada_erradamente': 'wrong_false_alarm_CL.shape[0] / len(Y)',
        'erros_de_rotulo_ajustados_corretamente': 'len(erros_ajustados_corretamente_CL)',
        'taxa_do_erro_ajustada_corretamente': 'len(erros_ajustados_corretamente_CL) / len(erros_indices_CL) if erros_indices_CL else 0',
        'taxa_do_erro_nao_corrigida': 'len(erros_nao_corrigidos_CL) / len(erros_indices_CL) if erros_indices_CL else 0',
        'novos_erros_gerados': 'len(novos_erros_gerados_CL) if erros_indices_CL else 0',
        'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': 'len(novos_erros_gerados_CL) / len(Y) if erros_indices_CL else 0',
    }
        
    with open(caminho_ocpc, "w") as f:
        json.dump(resultado_outliers_ocpc, f, indent=4)
        
    with open(caminho_cl, "w") as f:
        json.dump(resultado_outliers_CL, f, indent=4)
    
    return resultado_outliers_ocpc, resultado_outliers_CL