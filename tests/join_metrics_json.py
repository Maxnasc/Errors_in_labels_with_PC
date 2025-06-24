import json
import pandas as pd

def get_information(dataset):
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

if __name__=="__main__":
    # Importar os arquivos json de métricas para cada dataset e formar um único arquivo pra cada dataset
    breast = get_information('breast_cancer')
    iris = get_information('load_iris')
    wine = get_information('load_wine')
    sintetic_2D_dataset = get_information('sintetic_2D_dataset')
    
    df = pd.concat([breast, iris, wine, sintetic_2D_dataset], ignore_index=True)
    
    colunas_para_media = ['taxa_de_erro_detectada_corretamente',	'taxa_de_erro_detectada_erradamente',	'erros_de_rotulo_ajustados_corretamente',	'taxa_do_erro_ajustada_corretamente',	'taxa_do_erro_nao_corrigida',	'novos_erros_gerados',	'taxa_de_erro_novos_erros_gerados_com_relacao_ao_dataset_original']
    media_colunas = df[colunas_para_media].mean()
    df_media = pd.DataFrame(media_colunas).T
    df_media.index = ['Média']
    
    df = pd.concat([df, df_media])
    
    df.to_excel('resultados_pos_nsga_III.xlsx')


    # Fazer o mesmo para as emissões