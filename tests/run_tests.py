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
    def get_indices(values):
        result = {
            f"mean": np.mean([values]),
            f"variance": np.var(values),  # por padrão, populacional
            f"std": np.std(values)  # populacional
        }
        return result

    guide_metrics = {
        "ocpc": {
            'taxa_de_erro_detectada_corretamente': [],
            'taxa_de_erro_detectada_erradamente': [],
            'erros_de_rotulo_ajustados_corretamente': [],
            'taxa_do_erro_ajustada_corretamente': [],
            'taxa_do_erro_nao_corrigida': [],
            'novos_erros_gerados': [],
            'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': [],
        },
        "CL":{
            'taxa_de_erro_detectada_corretamente': [],
            'taxa_de_erro_detectada_erradamente': [],
            'erros_de_rotulo_ajustados_corretamente': [],
            'taxa_do_erro_ajustada_corretamente': [],
            'taxa_do_erro_nao_corrigida': [],
            'novos_erros_gerados': [],
            'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': [],
        }}
    
    result = {
        "ocpc": {
            'taxa_de_erro_detectada_corretamente': [],
            'taxa_de_erro_detectada_erradamente': [],
            'erros_de_rotulo_ajustados_corretamente': [],
            'taxa_do_erro_ajustada_corretamente': [],
            'taxa_do_erro_nao_corrigida': [],
            'novos_erros_gerados': [],
            'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': [],
        },
        "CL":{
            'taxa_de_erro_detectada_corretamente': [],
            'taxa_de_erro_detectada_erradamente': [],
            'erros_de_rotulo_ajustados_corretamente': [],
            'taxa_do_erro_ajustada_corretamente': [],
            'taxa_do_erro_nao_corrigida': [],
            'novos_erros_gerados': [],
            'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': [],
        }}
    
    for method, metrics in guide_metrics.items():
        for metric in metrics:
            dataset_info_compiled = [dataset_values[method][metric] for dataset_name, dataset_values in global_metrics.items()]
            result[method][metric] = get_indices(dataset_info_compiled)
            
    return result

def run_and_track_emissions(dataset_function, outlier_detection_ocpc: bool = None, method_name: str = "unknown"):
    tracker = EmissionsTracker(output_dir="./codecarbon_emissions", output_file=f"emissions_{method_name}.csv")
    tracker.start()
    if outlier_detection_ocpc is not None:
        metrics = dataset_function(outlier_detection_OCPC=outlier_detection_ocpc)
    else:
        metrics = dataset_function()
    tracker.stop()
    return metrics

def calculate_mean_of_samples(data: dict):
    aux_global = {}
    for dataset_name, dataset in data.items():
        aux_dataset = {
        "ocpc": {
            'taxa_de_erro_detectada_corretamente': [],
            'taxa_de_erro_detectada_erradamente': [],
            'erros_de_rotulo_ajustados_corretamente': [],
            'taxa_do_erro_ajustada_corretamente': [],
            'taxa_do_erro_nao_corrigida': [],
            'novos_erros_gerados': [],
            'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': [],
        },
        "CL":{
            'taxa_de_erro_detectada_corretamente': [],
            'taxa_de_erro_detectada_erradamente': [],
            'erros_de_rotulo_ajustados_corretamente': [],
            'taxa_do_erro_ajustada_corretamente': [],
            'taxa_do_erro_nao_corrigida': [],
            'novos_erros_gerados': [],
            'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original': [],
        }}
        for leitura in dataset:
            for metodo, resultados in leitura.items():
                for key, value in resultados.items():
                    aux_dataset[metodo][key].append(value)
        aux_global[dataset_name] = {"ocpc": {}, "CL": {}}
        for metodo, valores in aux_dataset.items():
            for key, list_values in valores.items():
                aux_global[dataset_name][metodo][key] = statistics.mean(list_values)
    return aux_global

def get_metrics_from_two_outlier_detection_method(n_samples: int):
    global_metrics_PC = {}
    global_metrics_LOF = {}

    # Executando os testes com PC_LabelCorrector (OCPC = True) e rastreando emissões
    print("Executando testes com PC_LabelCorrector...")
    # try:
    #     global_metrics_PC['metric_2D'] = [run_and_track_emissions(test_2D_sintetic_dataset, outlier_detection_ocpc=True, method_name="PC_2D") for i in range(n_samples)]
    # except Exception as e:
        # global_metrics_PC['metric_2D'] = {"Erro": str(e)}

    # try:
    #     global_metrics_PC['metric_breast_cancer'] = [run_and_track_emissions(test_breast_cancer_dataset, outlier_detection_ocpc=True, method_name="PC_breast_cancer") for i in range(n_samples)]
    # except Exception as e:
    #     global_metrics_PC['metric_breast_cancer'] = {"Erro": str(e)}

    # # try:
    # #     global_metrics_PC['metric_digits'] = run_and_track_emissions(test_digits_dataset, outlier_detection_ocpc=True, method_name="PC_digits")
    # # except Exception as e:
    # #     global_metrics_PC['metric_digits'] = {"Erro": str(e)}

    # # try:
    # #     global_metrics_PC['metric_linnerud'] = run_and_track_emissions(test_linnerud_dataset, outlier_detection_ocpc=True, method_name="PC_linnerud")
    # # except Exception as e:
    # #     global_metrics_PC['metric_linnerud'] = {"Erro": str(e)}

    # try:
    #     global_metrics_PC['metric_load_iris'] = [run_and_track_emissions(test_load_iris_dataset, outlier_detection_ocpc=True, method_name="PC_load_iris") for i in range(n_samples)]
    # except Exception as e:
    #     global_metrics_PC['metric_load_iris'] = {"Erro": str(e)}

    # try:
    #     global_metrics_PC['metric_load_wine'] = [run_and_track_emissions(test_load_wine_dataset, outlier_detection_ocpc=True, method_name="PC_load_wine") for i in range(n_samples)]
    # except Exception as e:
    #     global_metrics_PC['metric_load_wine'] = {"Erro": str(e)}

    # global_metrics_PC = get_statistics(calculate_mean_of_samples(global_metrics_PC), '_OCPC')

    #####################################################################

    # # Executando os testes com LOF (OCPC = False) e rastreando emissões
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

    global_metrics_LOF = get_statistics(calculate_mean_of_samples(global_metrics_LOF), '_CL')

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
        # linhas_csv = []
        # for chave_externa, dicionario_interno in data.items():
        #     linha = {'Métrica': chave_externa}  # Primeira coluna com a chave externa
        #     if isinstance(dicionario_interno, dict):
        #         linha.update(dicionario_interno)  # Adiciona as chaves e valores do dicionário interno
        #     else:
        #         linha['Valor'] = dicionario_interno # Se o valor não for um dicionário, coloca em uma coluna 'Valor'
        #     linhas_csv.append(linha)

        # dataframe = pd.DataFrame(linhas_csv)
        # dataframe.to_csv(nome_arquivo_csv, index=False, encoding='utf-8')
        # print(f"Arquivo CSV '{nome_arquivo_csv}' criado com sucesso.")
        registros = []
        for modelo, metricas in data.items():
            for metrica, casos in metricas.items():
                for caso, valor in casos.items():
                    registros.append({
                        'modelo': modelo,
                        'caso': caso,
                        'metrica': metrica,
                        'valor': valor
                    })

        # Etapa 2: criar o DataFrame
        df = pd.DataFrame(registros)

        # Etapa 3 (opcional): pivotar se quiser "modelo + caso" como índice e métricas como colunas
        df_pivot = df.pivot_table(
            index=['modelo', 'caso'],
            columns='metrica',
            values='valor'
        ).reset_index()

        df_pivot.to_excel('tests/correcoes_resultantes.xlsx')

    # dict_to_csv_file(nome_arquivo_csv=f'{path}_PC.csv', data=global_metrics_PC)
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
    
    # get_emmisions_metric()

def get_emmisions_metric():
    paths = 'tests/breast_cancer/codecarbon_emissions'
    folder_paths = 'breast_cancer', 'load_iris', 'load_wine', 'sintetic_2D_dataset'
    
    df_emissions_ocpc = pd.DataFrame()
    df_emissions_cl = pd.DataFrame()
    
    for f_path in folder_paths:
        df_ocpc = pd.read_csv(f'tests/{f_path}/codecarbon_emissions/emissions_PC_2D_sintetic_OCPC.csv')
        df_cl = pd.read_csv(f'tests/{f_path}/codecarbon_emissions/emissions_CL_2D_sintetic_OCPC.csv')
        
        df_emissions_ocpc = pd.concat([df_emissions_ocpc, df_ocpc], ignore_index=True)
        df_emissions_cl = pd.concat([df_emissions_cl, df_cl], ignore_index=True)
        
    colunas = ['duration', 'emissions', 'energy_consumed']
    # Remove linhas com NaNs nas colunas desejadas
    df_limpo = df_emissions_ocpc.dropna(subset=colunas)
    # Converte essas colunas para tipo numérico (coercivo: transforma strings inválidas em NaN)
    df_limpo[colunas] = df_limpo[colunas].apply(pd.to_numeric, errors='coerce')
    # Remove quaisquer novos NaNs gerados pela conversão
    df_limpo = df_limpo.dropna(subset=colunas)
    # Calcula a média e transforma em linha de DataFrame
    df_ocpc = df_limpo[colunas].mean().to_frame().T
    df_ocpc.insert(loc=0, column='metodo', value='ocpc')    
    
    
    # Remove linhas com NaNs nas colunas desejadas
    df_limpo_cl = df_emissions_cl.dropna(subset=colunas)
    # Converte essas colunas para tipo numérico (coercivo: transforma strings inválidas em NaN)
    df_limpo_cl[colunas] = df_limpo_cl[colunas].apply(pd.to_numeric, errors='coerce')
    # Remove quaisquer novos NaNs gerados pela conversão
    df_limpo_cl = df_limpo_cl.dropna(subset=colunas)
    # Calcula a média e transforma em linha de DataFrame
    df_cl = df_limpo_cl[colunas].mean().to_frame().T
    df_cl.insert(loc=0, column='metodo', value='CL')
    
    df_emissions = pd.concat([df_ocpc, df_cl], ignore_index=True)
    df_emissions.to_excel('tests/emissoes_resultantes.xlsx')

if __name__=="__main__":
    get_metrics_from_two_outlier_detection_method(n_samples=1)