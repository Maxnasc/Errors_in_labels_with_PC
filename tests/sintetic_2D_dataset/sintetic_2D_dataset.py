import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import calcula_novas_metricas, get_dataset_with_error, save_metrics_to_csv_file, save_metrics_to_json_file
import os
from codecarbon import EmissionsTracker

def run_label_correction(data, target, outlier_detection_ocpc: bool, tracker_prefix: str, k_max: int, alfa: float, lamda: float, f: float):
    tracker = EmissionsTracker(output_dir="tests/sintetic_2D_dataset/codecarbon_emissions", output_file=f"emissions_{tracker_prefix}.csv")
    tracker.start()
    # lc = PC_LabelCorrector(path='sintetic_2D_dataset', detect_outlier_with_ocpc=outlier_detection_ocpc, k_max=k_max, alfa=alfa, lamda=lamda, f=f)
    lc = PC_LabelCorrector(path='sintetic_2D_dataset', detect_outlier_with_ocpc=outlier_detection_ocpc)
    Y_adjusted = lc.run(X=data, Y=target)
    tracker.stop()
    return Y_adjusted, lc.metrics

def run_confident_learning(data, target, original_target, tracker_prefix: str):
    tracker = EmissionsTracker(output_dir="tests/sintetic_2D_dataset/codecarbon_emissions", output_file=f"emissions_{tracker_prefix}.csv")
    tracker.start()
    cl_issues, issues = get_CL_label_correction(data, target, original_target)
    tracker.stop()
    return cl_issues, issues

def plot_outliers(X, Y, data_with_error):
    # Supondo que X, Y e data_with_error já estejam definidos
    X_plot = X  # (202, 2)
    Y_original = Y
    Y_com_erro = data_with_error.get("target")

    # Identificar índices
    indices_com_erro = [i for i in range(len(Y_original)) if Y_original[i] != Y_com_erro[i]]
    indices_sem_erro = [i for i in range(len(Y_original)) if Y_original[i] == Y_com_erro[i]]

    # Criação do gráfico
    plt.figure(figsize=(8, 6))

    # Círculos para rótulos corretos (coloridos conforme Y_com_erro)
    scatter = plt.scatter(
        X_plot[indices_sem_erro, 0],
        X_plot[indices_sem_erro, 1],
        c=Y_com_erro[indices_sem_erro],
        cmap='viridis',
        marker='o',
        label='Rótulo correto'
    )

    # Estrelas para rótulos com erro (coloridas conforme Y_com_erro)
    plt.scatter(
        X_plot[indices_com_erro, 0],
        X_plot[indices_com_erro, 1],
        c=Y_com_erro[indices_com_erro],
        cmap='viridis',
        marker='*',
        s=150,
        label='Rótulo com erro'
    )

    # Barra de cores
    plt.colorbar(scatter, label='Valor do rótulo com erro (Y_com_erro)')

    plt.title('Visualização dos dados: círculos corretos, estrelas com erro (coloridas)')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def test_2D_sintetic_dataset(path = 'sintetic_2D_dataset', k_max = int, alfa = float, lamda = float, f = float, outlier_detection_OCPC=True):
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
    
    # plot_outliers(X, Y, data_with_error) # PLOTA OUTLIERS

    labels_wrong_before = sum(1 for i in range(len(Y)) if data_with_error["target"][i] != Y[i])
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
        Y,
        f"CL_2D_sintetic_{'OCPC' if outlier_detection_OCPC else 'LOF'}"
    )
    
    # # entender quais labels estão errados
    # erros = [i for i, value in enumerate(Y) if value != data_with_error.get('target')[i]]
    
    # if outlier_detection_OCPC:
    #     caminho_ouliers_pc = 'tests/sintetic_2D_dataset/outliers_ocpc.json'
    #     caminho_ocpc = 'tests/sintetic_2D_dataset/outliers_result_ocpc.json'
    # else:
    #     caminho_ouliers_pc = 'tests/sintetic_2D_dataset/outliers_lof.json'
    #     caminho_ocpc = 'tests/sintetic_2D_dataset/outliers_result_lof.json'
    
    # with open(caminho_ouliers_pc, 'r') as opcl:
    #     outliers_saved = json.load(opcl)    
        
    # correct_detected_outliers = [o for i, o in enumerate(outliers_saved) if o == -1 and i in erros]
    # wrong_detected_outliers = [o for i, o in enumerate(outliers_saved) if o == -1 and i not in erros]
    
    # # Qual a porcentagem do erro que foi efetivamente corrigida ?
    # erros_depois_de_corrigir = [i for i, value in enumerate(Y) if value != Y_adjusted_pc[i]]
    
    # erros_ajustados_corretamente = [i for i in erros if i not in erros_depois_de_corrigir]
    # erros_nao_corrigidos = [i for i in erros if i in erros_depois_de_corrigir]
    # novos_erros_gerados = [i for i in erros_depois_de_corrigir if i not in erros]  

    # # Cálculos para resultado_outliers_CL, espelhando a lógica do OCPC
    # issues['erro_original'] = issues['original_labels'] != data_with_error.get('target')
    # issues['erro_apos_correcao'] = issues['original_labels'] != issues['predicted_label']

    # erros_indices_CL = issues[issues['erro_original']].index.tolist()
    # erros_depois_corrigir_indices_CL = issues[issues['erro_apos_correcao']].index.tolist()

    # erros_ajustados_corretamente_CL = [i for i in erros_indices_CL if i not in erros_depois_corrigir_indices_CL]
    # erros_nao_corrigidos_CL = [i for i in erros_indices_CL if i in erros_depois_corrigir_indices_CL]
    # novos_erros_gerados_CL = [i for i in erros_depois_corrigir_indices_CL if i not in erros_indices_CL]

    # correct_outliers_detected_CL = issues[issues['is_label_issue'] & (issues['given_label'] == issues['original_labels'])]
    # wrong_false_alarm_CL = issues[issues['is_label_issue'] & (issues['given_label'] != issues['original_labels'])]
    
    # resultado_outliers_ocpc = {
    #     'taxa_de_erro_detectada_corretamente': len(correct_detected_outliers)/len(Y),
    #     'taxa_de_erro_detectada_erradamente': len(wrong_detected_outliers)/len(Y),
    #     'erros_de_rotulo_ajustados_corretamente': len(erros_ajustados_corretamente),
    #     'taxa_do_erro_ajustada_corretamente': len(erros_ajustados_corretamente)/len(erros),
    #     'taxa_do_erro_nao_corrigida': len(erros_nao_corrigidos)/len(erros),
    #     'novos_erros_gerados': len(novos_erros_gerados)/len(erros),
    #     'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': len(novos_erros_gerados)/len(Y),
    # }
    
    # resultado_outliers_CL = {
    #     'taxa_de_erro_detectada_corretamente': correct_outliers_detected_CL.shape[0] / len(Y),
    #     'taxa_de_erro_detectada_erradamente': wrong_false_alarm_CL.shape[0] / len(Y),
    #     'erros_de_rotulo_ajustados_corretamente': len(erros_ajustados_corretamente_CL),
    #     'taxa_do_erro_ajustada_corretamente': len(erros_ajustados_corretamente_CL) / len(erros_indices_CL) if erros_indices_CL else 0,
    #     'taxa_do_erro_nao_corrigida': len(erros_nao_corrigidos_CL) / len(erros_indices_CL) if erros_indices_CL else 0,
    #     'novos_erros_gerados': len(novos_erros_gerados_CL) if erros_indices_CL else 0,
    #     'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': len(novos_erros_gerados_CL) / len(Y) if erros_indices_CL else 0,
    # }
        
    # with open(caminho_ocpc, "w") as f:
    #     json.dump(resultado_outliers_ocpc, f, indent=4)
        
    # with open('tests/sintetic_2D_dataset/outliers_result_cl.json', "w") as f:
    #     json.dump(resultado_outliers_CL, f, indent=4)
    
    resultado_outliers_ocpc, resultado_outliers_CL = calcula_novas_metricas(path=path, outlier_detection_OCPC=outlier_detection_OCPC, Y=Y, data_with_error=data_with_error, Y_adjusted_pc=Y_adjusted_pc, issues=issues)

    metrics = {
        "ocpc": resultado_outliers_ocpc,
        "CL": resultado_outliers_CL
    }
    # metrics = {"original error rate PC_LabelCorrection": metrics_pc['original error rate']} | {"error rate after correction PC_LabelCorrection": metrics_pc['error rate after correction']} | cl_issues

    path='tests/sintetic_2D_dataset/comparation'
    save_metrics_to_json_file(path=path, metrics=metrics)

    return metrics

if __name__ == "__main__":
    test_2D_sintetic_dataset(path='sintetic_2D_dataset', outlier_detection_OCPC=True)
    test_2D_sintetic_dataset(path='sintetic_2D_dataset', outlier_detection_OCPC=False)