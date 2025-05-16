import pandas as pd

# Preciso baixar todos os documentos de cada banco de dados
# Preciso baixar todos os global_metrics.xlsx


# Baixando os dados de métrica das configs original, 01, 02 e 06 (mais relevantes)
df_config_orig = pd.read_excel('tests/global_metrics_config_orig.xlsx')
df_config_01 = pd.read_excel('tests/global_metrics_config_01.xlsx')
df_config_02 = pd.read_excel('tests/global_metrics_config_02.xlsx')
df_config_06 = pd.read_excel('tests/global_metrics_config_06.xlsx')

lines_PC_mask = [4, 6]
lines_CL_mask = [5, 7]
colums_mask = ['before_fix_mean', 'before_fix_variance', 'before_fix_std', 'after_fix_mean', 'after_fix_variance', 'after_fix_std']

df_PC_orig = df_config_orig.loc[lines_PC_mask, colums_mask]
df_CL_orig = df_config_orig.loc[lines_CL_mask, colums_mask]
df_PC_01 = df_config_01.loc[lines_PC_mask, colums_mask]
df_CL_01 = df_config_01.loc[lines_CL_mask, colums_mask]
df_PC_02 = df_config_02.loc[lines_PC_mask, colums_mask]
df_CL_02 = df_config_02.loc[lines_CL_mask, colums_mask]
df_PC_06 = df_config_06.loc[lines_PC_mask, colums_mask]
df_CL_06 = df_config_06.loc[lines_CL_mask, colums_mask]

dfs_pc = {'original': df_PC_orig, 'config_01': df_PC_01, 'config_02': df_PC_02, 'config_06': df_PC_06}
dfs_cl = {'original': df_CL_orig, 'config_01': df_CL_01, 'config_02': df_CL_02, 'config_06': df_CL_06}

def get_after_fix_mean(df):
    """Retorna o valor de after_fix_mean da primeira linha não-NaN."""
    not_na_df = df[df['after_fix_mean'].notna()]
    if not not_na_df.empty:
        return not_na_df['after_fix_mean'].iloc[0]
    return None

# Encontrar a melhor configuração para PC
best_config_pc = None
min_mean_pc = float('inf')
pc_means = {}

for name, df in dfs_pc.items():
    mean_val = get_after_fix_mean(df)
    pc_means[name] = mean_val
    if mean_val is not None and mean_val < min_mean_pc:
        min_mean_pc = mean_val
        best_config_pc = name

print(f"Melhor configuração para PC (menor after_fix_mean): {best_config_pc} com valor {min_mean_pc}")

# Encontrar a melhor configuração para CL
best_config_cl = None
min_mean_cl = float('inf')
cl_means = {}

for name, df in dfs_cl.items():
    mean_val = get_after_fix_mean(df)
    cl_means[name] = mean_val
    if mean_val is not None and mean_val < min_mean_cl:
        min_mean_cl = mean_val
        best_config_cl = name

print(f"Melhor configuração para CL (menor after_fix_mean): {best_config_cl} com valor {min_mean_cl}")

# Criar DataFrame consolidado
all_data = {}
for name, df in dfs_pc.items():
    # Pegando a linha com before_fix (índice 0 após o filtro)
    before_fix_values = df[df['before_fix_mean'].notna()].iloc[0].rename(f'PC_{name}_before_fix')
    all_data[f'PC_{name}_before_fix'] = before_fix_values

    # Pegando a linha com after_fix (índice 1 após o filtro)
    after_fix_values = df[df['after_fix_mean'].notna()].iloc[0].rename(f'PC_{name}_after_fix')
    all_data[f'PC_{name}_after_fix'] = after_fix_values

for name, df in dfs_cl.items():
    # Pegando a linha com before_fix (índice 0 após o filtro)
    before_fix_values = df[df['before_fix_mean'].notna()].iloc[0].rename(f'CL_{name}_before_fix')
    all_data[f'CL_{name}_before_fix'] = before_fix_values

    # Pegando a linha com after_fix (índice 1 após o filtro)
    after_fix_values = df[df['after_fix_mean'].notna()].iloc[0].rename(f'CL_{name}_after_fix')
    all_data[f'CL_{name}_after_fix'] = after_fix_values

df_combinado = pd.DataFrame(all_data).T  # Transpondo para ter as configurações como índice

print("\nDataFrame Combinado:")
print(df_combinado)

df_combinado.to_csv('tests/comparation_between_configs/comparacao_metricas_configuracoes_parametros.csv')