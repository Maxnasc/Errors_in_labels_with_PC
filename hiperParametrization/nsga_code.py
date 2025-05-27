import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, cpu_count, freeze_support
import os
from yaspin import yaspin
from yaspin.spinners import Spinners
import time

# Assumindo que essas importações são para suas funções auxiliares
# Certifique-se de que o caminho para utils.py e PC_LabelCorrector.py está correto
from utils.utils import get_dataset_with_error
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector


# 1. Modificar o creator para Múltiplos Objetivos
# Objetivo 1: Minimizar o erro (peso -1.0)
# Objetivo 2: Minimizar o gasto energético (peso -1.0)
if "FitnessMultiObj" not in creator.__dict__:
    creator.create("FitnessMultiObj", base.Fitness, weights=(-1.0, -1.0))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMultiObj)

# Espaço de busca para os indivíduos (parâmetros do PC_LabelCorrector)
toolbox = base.Toolbox()
toolbox.register("attr_k_max", random.randint, 2, 10)
toolbox.register("attr_alfa", random.uniform, 0.1, 1.0)
toolbox.register("attr_lamda", random.uniform, 0.1, 1.0)
toolbox.register("attr_f", random.uniform, 0.5, 1.5)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_k_max, toolbox.attr_alfa, toolbox.attr_lamda, toolbox.attr_f),
                 n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Carregar datasets uma única vez fora da função de avaliação para otimização
erro_proposto = 0.1
datasets = {
    "iris": get_dataset_with_error(load_iris().data, load_iris().target, erro_proposto),
    "breast_cancer": get_dataset_with_error(load_breast_cancer().data, load_breast_cancer().target, erro_proposto),
    "wine": get_dataset_with_error(load_wine().data, load_wine().target, erro_proposto)
}

# 2. Função de avaliação adaptada para Múltiplos Objetivos
# Agora retorna uma tupla (erro, custo_energetico)
def evaluate_individual(individual):
    try:
        params = {
            'k_max': int(individual[0]),
            'alfa': float(individual[1]),
            'lamda': float(individual[2]),
            'f': float(individual[3]),
            'detect_outlier_with_ocpc': True
        }

        total_error = 0.0
        total_energy_cost = 0.0 # Inicializa o custo energético

        lc = PC_LabelCorrector(**params)

        for data_name, data in datasets.items():
            start_time = time.perf_counter() # Inicia a contagem de tempo
            lc.run(X=data["data"], Y=data["target"])
            end_time = time.perf_counter()   # Finaliza a contagem de tempo
            
            total_error += lc.metrics['error rate after correction']
            total_energy_cost += (end_time - start_time) # Acumula o tempo de execução como custo

        # Retorna uma tupla com os dois objetivos
        # Ambos serão minimizados, por isso os pesos (-1.0, -1.0) no FitnessMultiObj
        return (total_error / len(datasets), total_energy_cost)
    except Exception as e:
        # Penaliza indivíduos que causam erros com valores muito altos
        print(f"Erro na avaliação do indivíduo {individual}: {e}")
        return (1.0, 1000.0) # Retorne valores altos para penalizar (erro máximo 1.0, tempo alto)

# Função de mutação com verificação de limites (mantida igual)
def mutate_with_limits(individual, mu, sigma, indpb):
    mutated_individual = tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb)[0]
    # Garantir que os valores estejam dentro dos limites após a mutação
    mutated_individual[0] = max(2, min(10, int(round(mutated_individual[0]))))    # k_max
    mutated_individual[1] = max(0.1, min(1.0, mutated_individual[1]))             # alfa
    mutated_individual[2] = max(0.1, min(1.0, mutated_individual[2]))             # lamda
    mutated_individual[3] = max(0.5, min(1.5, mutated_individual[3]))             # f
    return mutated_individual,

# Configuração do GA (operações genéticas)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_with_limits, mu=0, sigma=0.1, indpb=0.2)

# 3. Mudar o Algoritmo de Seleção para NSGA-II
toolbox.register("select", tools.selNSGA2) # Use selNSGA2 para seleção multiobjetivo
toolbox.register("evaluate", evaluate_individual) # Registrar a função de avaliação na toolbox

def run_nsga2_parallel(seed=None):
    random.seed(seed)
    np.random.seed(seed)

    pop_size = 100
    generations = 100
    cxpb, mutpb = 0.9, 0.1 # Probabilidades de cruzamento e mutação

    population = toolbox.population(n=pop_size)

    # Ferramentas para coletar estatísticas do algoritmo
    # stats.register("avg", np.mean, axis=0)  # Média de cada objetivo
    # stats.register("std", np.std, axis=0)   # Desvio padrão de cada objetivo
    # stats.register("min", np.min, axis=0)   # Mínimo de cada objetivo
    # stats.register("max", np.max, axis=0)   # Máximo de cada objetivo
    # logbook = tools.Logbook()
    # logbook.header = "gen", "evals", "std", "min", "avg", "max" # Opcional: para log detalhado da evolução

    with yaspin(Spinners.dots, text="Rodando gerações do NSGA-II...") as sp:
        # Avaliar a população inicial
        # Use o pool para a avaliação da população inicial
        # fitnesses = list(map(toolbox.evaluate, population)) # Isso não usaria o pool
        # Para usar o pool, o map deve ser configurado na toolbox:
        # toolbox.register("map", pool.map) # Isso é feito fora, no main, para o Pool
        
        # O algorithms.eaMuPlusLambda já usa a função map registrada na toolbox,
        # que será o map do Pool.
        
        # algorithms.eaMuPlusLambda é um algoritmo de evolução (como o eaSimple)
        # que implementa o paradigma (mu + lambda) onde 'mu' pais e 'lambda' filhos
        # competem para formar a próxima geração. NSGA-II é a estratégia de seleção.
        population, logbook = algorithms.eaMuPlusLambda(population, toolbox, mu=pop_size, # Número de pais
                                                        lambda_=pop_size, # Número de filhos gerados
                                                        cxpb=cxpb, mutpb=mutpb, ngen=generations,
                                                        stats=None, halloffame=None, verbose=False) # stats e verbose podem ser True para logs

        sp.text = "Gerações do NSGA-II concluídas!"
        sp.ok("✅")

    # A fronteira de Pareto final são os indivíduos não dominados da última população
    # `k=pop_size` e `first_front_only=True` garante que você pegue a primeira frente de Pareto.
    pareto_front = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
    
    # Retorna a fronteira de Pareto para esta execução
    return pareto_front

if __name__ == "__main__":
    freeze_support() # Necessário para o multiprocessing em Windows

    total_runs = 5 # Número de vezes que o NSGA-II será executado. Aumente para mais robustez.
    all_pareto_points_from_all_runs = [] # Para armazenar todos os indivíduos da fronteira de Pareto de todas as runs

    # Configuração do pool de processos
    # Isso precisa ser feito ANTES de registrar o map na toolbox
    # para que as funções do pool possam ser serializadas corretamente
    with Pool(processes=max(1, cpu_count()-1)) as pool:
        toolbox.register("map", pool.map) # Registra o map do Pool na toolbox para avaliações paralelas

        for run_idx in tqdm(range(total_runs), desc="Executando runs do NSGA-II"):
            # O NSGA-II retorna a fronteira de Pareto para cada run
            current_pareto_front = run_nsga2_parallel(seed=run_idx) # Passe uma seed diferente para cada run
            
            # Adicione os indivíduos da fronteira de Pareto desta run à lista geral
            for ind in current_pareto_front:
                all_pareto_points_from_all_runs.append({
                    'k_max': int(ind[0]),
                    'alfa': float(ind[1]),
                    'lamda': float(ind[2]),
                    'f': float(ind[3]),
                    'error_rate': ind.fitness.values[0], # Primeiro objetivo
                    'energy_cost': ind.fitness.values[1] # Segundo objetivo
                })

    # Salvar todos os indivíduos que fizeram parte das fronteiras de Pareto encontradas
    df_results_mo = pd.DataFrame(all_pareto_points_from_all_runs)
    df_results_mo.to_csv("resultados_otimizacao_ga_multiobjetivo.csv", index=False)

    print("\n✅ Otimização multiobjetivo concluída!")
    print(f"Resultados salvos em 'resultados_otimizacao_ga_multiobjetivo.csv'.")

    # --- Plotando a Fronteira de Pareto Agregada ---
    # Para visualizar a fronteira de Pareto aproximada encontrada por TODAS as runs.
    # É útil para ver a distribuição das soluções não dominadas.
    
    # Se houver resultados, plota
    if not df_results_mo.empty:
        plt.figure(figsize=(10, 8))
        plt.scatter(df_results_mo['error_rate'], df_results_mo['energy_cost'], 
                    c='blue', alpha=0.6, label='Indivíduos da Fronteira de Pareto')
        
        plt.title("Fronteira de Pareto Aproximada (Erro vs. Gasto Energético)")
        plt.xlabel("Taxa de Erro Média (Minimizar)")
        plt.ylabel("Gasto Energético (Minimizar - tempo em segundos)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("fronteira_pareto_multiobjetivo.png")
        plt.show()
    else:
        print("Nenhum resultado para plotar a fronteira de Pareto.")

    # Exemplo de como você poderia selecionar uma solução "ideal" da fronteira
    # Por exemplo, a solução com menor erro entre as de baixo custo energético,
    # ou a solução com menor custo energético entre as de erro aceitável.
    
    # Se df_results_mo não estiver vazio:
    if not df_results_mo.empty:
        # Exemplo: Selecionar o indivíduo que tem o menor erro na fronteira de Pareto
        best_error_solution = df_results_mo.loc[df_results_mo['error_rate'].idxmin()]
        print("\nSolução com menor Taxa de Erro na Fronteira de Pareto:")
        print(best_error_solution)

        # Exemplo: Selecionar o indivíduo com menor Gasto Energético na fronteira de Pareto
        best_energy_solution = df_results_mo.loc[df_results_mo['energy_cost'].idxmin()]
        print("\nSolução com menor Gasto Energético na Fronteira de Pareto:")
        print(best_energy_solution)

        # Você pode usar outras métricas para avaliar a qualidade da fronteira
        # ou escolher a solução preferida.
        # Por exemplo, encontrar o ponto que está mais próximo da origem (0,0)
        # se os objetivos fossem minimizados em valor absoluto e tivessem a mesma escala.
        # df_results_mo['distance_to_origin'] = np.sqrt(df_results_mo['error_rate']**2 + df_results_mo['energy_cost']**2)
        # best_compromise_solution = df_results_mo.loc[df_results_mo['distance_to_origin'].idxmin()]
        # print("\nSolução de Compromisso (mais próxima da origem):")
        # print(best_compromise_solution)