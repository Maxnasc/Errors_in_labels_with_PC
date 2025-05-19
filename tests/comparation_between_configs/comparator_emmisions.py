import pandas as pd
import statistics

folder_names = ['orig', '01', '02', '06', '07', '08']
CL_file_names = ['CL_2D', 'CL_breast_cancer', 'CL_load_iris', 'CL_load_wine']
PC_file_names = ['PC_2D', 'PC_breast_cancer', 'PC_load_iris', 'PC_load_wine']

emissions_PC = {}
emissions_CL = {}

for folder_name in folder_names:
    # Processamento para PC
    emissions_PC_aux = {
        'duration': [],
        'emissions': [],
        'energy_consumed': []
    }
    for file_name in PC_file_names:
        try:
            df = pd.read_csv(f'codecarbon_emissions_config_{folder_name}/emissions_{file_name}.csv')
            emissions_PC_aux['duration'].append(df['duration'].mean())
            emissions_PC_aux['emissions'].append(df['emissions'].mean())
            emissions_PC_aux['energy_consumed'].append(df['energy_consumed'].mean())
        except FileNotFoundError:
            print(f"Arquivo não encontrado: codecarbon_emissions_config_{folder_name}/emissions_{file_name}.csv")
            emissions_PC_aux['duration'].append(None)
            emissions_PC_aux['emissions'].append(None)
            emissions_PC_aux['energy_consumed'].append(None)

    emissions_PC[folder_name] = {
        'duration': statistics.mean([val for val in emissions_PC_aux['duration'] if val is not None]),
        'emissions': statistics.mean([val for val in emissions_PC_aux['emissions'] if val is not None]),
        'energy_consumed': statistics.mean([val for val in emissions_PC_aux['energy_consumed'] if val is not None])
    }

    # Processamento para CL
    emissions_CL_aux = {
        'duration': [],
        'emissions': [],
        'energy_consumed': []
    }
    for file_name in CL_file_names:
        try:
            df = pd.read_csv(f'codecarbon_emissions_config_{folder_name}/emissions_{file_name}.csv')
            emissions_CL_aux['duration'].append(df['duration'].mean())
            emissions_CL_aux['emissions'].append(df['emissions'].mean())
            emissions_CL_aux['energy_consumed'].append(df['energy_consumed'].mean())
        except FileNotFoundError:
            print(f"Arquivo não encontrado: codecarbon_emissions_config_{folder_name}/emissions_{file_name}.csv")
            emissions_CL_aux['duration'].append(None)
            emissions_CL_aux['emissions'].append(None)
            emissions_CL_aux['energy_consumed'].append(None)

    emissions_CL[folder_name] = {
        'duration': statistics.mean([val for val in emissions_CL_aux['duration'] if val is not None]),
        'emissions': statistics.mean([val for val in emissions_CL_aux['emissions'] if val is not None]),
        'energy_consumed': statistics.mean([val for val in emissions_CL_aux['energy_consumed'] if val is not None])
    }

cl = pd.DataFrame(emissions_CL)
pc = pd.DataFrame(emissions_PC)

cl.to_csv('tests/comparation_between_configs/comparacao_emissoes_configuracoes_parametros_CL.csv')
pc.to_csv('tests/comparation_between_configs/comparacao_emissoes_configuracoes_parametros_PC.csv')