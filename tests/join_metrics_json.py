import json
import pandas as pd

def get_correction_information(dataset):
    cl = f'tests/{dataset}/outliers_result_cl.json'
    lof = f'tests/{dataset}/outliers_result_lof.json'
    
    with open(cl, 'r') as file:
        cl_json = json.load(file)
        
    with open(lof, 'r') as lofFile:
        lof_json = json.load(lofFile)
        
    combined = {
        'cl': cl_json,
        'lof': lof_json
    }
    
    df_combined = pd.DataFrame.from_dict(combined, orient='index')
    df_combined = df_combined.reset_index()
    df_combined = df_combined.rename(columns={'index':'metodo'})
    df_combined['dataset'] = dataset
    
    colunas = df_combined.columns.to_list()
    
    colunas.remove('dataset')
    colunas.insert(0, 'dataset')
    
    df_combined = df_combined[colunas]

    return df_combined

def get_emission_information(dataset):
    cl_df = pd.read_csv(f'tests/{dataset}/codecarbon_emissions/emissions_CL_2D_sintetic_LOF.csv')
    lof_df = pd.read_csv(f'tests/{dataset}/codecarbon_emissions/emissions_PC_2D_sintetic_LOF.csv')
    
    df_combined = pd.concat([cl_df, lof_df], ignore_index=True)
    df_combined = df_combined.reset_index()
    df_combined['dataset'] = dataset
    
    colunas = df_combined.columns.to_list()
    
    colunas.remove('dataset')
    colunas.insert(0, 'dataset')
    
    df_combined = df_combined[colunas]

    return df_combined

if __name__=="__main__":
    # Importar os arquivos json de métricas para cada dataset e formar um único arquivo pra cada dataset
    breast = get_correction_information('breast_cancer')
    iris = get_correction_information('load_iris')
    wine = get_correction_information('load_wine')
    sintetic_2D_dataset = get_correction_information('sintetic_2D_dataset')
    
    df = pd.concat([breast, iris, wine, sintetic_2D_dataset], ignore_index=True)
    
    colunas_para_media = ['taxa_de_erro_detectada_corretamente',	'taxa_de_erro_detectada_erradamente',	'erros_de_rotulo_ajustados_corretamente',	'taxa_do_erro_ajustada_corretamente',	'taxa_do_erro_nao_corrigida',	'novos_erros_gerados',	'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original']
    media_colunas = df[colunas_para_media].mean()
    df_media = pd.DataFrame(media_colunas).T
    df_media.index = ['Média']
    
    df = pd.concat([df, df_media])
    
    df.to_excel('resultados_pos_nsga_V.xlsx')

    e_breast = get_emission_information('breast_cancer')
    e_iris = get_emission_information('load_iris')
    e_wine = get_emission_information('load_wine')
    e_sintetic_2D_dataset = get_emission_information('sintetic_2D_dataset')
    
    df = pd.concat([e_breast, e_iris, e_wine, e_sintetic_2D_dataset], ignore_index=True)
    
    colunas_para_media = ['taxa_de_erro_detectada_corretamente',	'taxa_de_erro_detectada_erradamente',	'erros_de_rotulo_ajustados_corretamente',	'taxa_do_erro_ajustada_corretamente',	'taxa_do_erro_nao_corrigida',	'novos_erros_gerados',	'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original']
    media_colunas = df[colunas_para_media].mean()
    df_media = pd.DataFrame(media_colunas).T
    df_media.index = ['Média']
    
    df = pd.concat([df, df_media])
    
    df.to_excel('resultados_pos_nsga_V.xlsx')
    
    # Fazer o mesmo para as emissões