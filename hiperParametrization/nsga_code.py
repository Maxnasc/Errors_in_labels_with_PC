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

# Importar as quatro funções de teste (que já instanciam internamente o PC_LabelCorrector)
from tests.breast_cancer.breast_cancer import test_breast_cancer_dataset
from tests.load_iris.load_iris import test_load_iris_dataset
from tests.load_wine.load_wine import test_load_wine_dataset
from tests.sintetic_2D_dataset.sintetic_2D_dataset import test_2D_sintetic_dataset

# -----------------------------------------------------------------------------
# 1. Modificar o creator para Múltiplos Objetivos (agora 5 objetivos)
#     todos os cinco devem ser minimizados, portanto os pesos são todos -1.0
if "FitnessMultiObj" not in creator.__dict__:
    creator.create("FitnessMultiObj", base.Fitness,
                   weights=(-1.0, -1.0, -1.0, -1.0, -1.0))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMultiObj)

# -----------------------------------------------------------------------------
# Espaço de busca para os indivíduos (quatro hiperparâmetros do PC_LabelCorrector)
toolbox = base.Toolbox()
toolbox.register("attr_k_max", random.randint, 2, 10)
toolbox.register("attr_alfa", random.uniform, 0.1, 1.0)
toolbox.register("attr_lamda", random.uniform, 0.1, 1.0)
toolbox.register("attr_f", random.uniform, 0.5, 1.5)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_k_max,
                  toolbox.attr_alfa,
                  toolbox.attr_lamda,
                  toolbox.attr_f),
                 n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# -----------------------------------------------------------------------------
# Lista das quatro funções de teste e seus “paths” correspondentes:
test_functions = [
    (test_breast_cancer_dataset,         "breast_cancer"),
    (test_load_iris_dataset,             "load_iris"),
    (test_load_wine_dataset,             "load_wine"),
    (test_2D_sintetic_dataset,           "sintetic_2D_dataset")
]

# -----------------------------------------------------------------------------
# 2. Função de avaliação adaptada para 5 objetivos
def evaluate_individual(individual):
    """
    Cada indivíduo é [k_max, alfa, lamda, f]. Chamamos as quatro funções de teste,
    medimos o tempo de cada execução e extraímos, do dicionário retornado,
    as métricas ocpc[…]. Por fim, montamos um vetor-fitness com 5 valores:
        1) | correct_detected_outliers_rate – 0.1 |
        2) wrong_detected_outliers_rate
        3) | correct_adjusted_errors_pc_rate – 0.1 |
        4) wrong_adjusted_errors_pc_rate
        5) total_time (soma de todos os tempos de cada teste)
    """
    try:
        # Extrair os hiperparâmetros do indivíduo
        k_max = int(individual[0])
        alfa = float(individual[1])
        lamda = float(individual[2])
        f = float(individual[3])

        # Somas acumuladas das métricas (para depois tirar média)
        sum_cdor = 0.0  # correct_detected_outliers_rate
        sum_wdor = 0.0  # wrong_detected_outliers_rate
        sum_caer = 0.0  # correct_adjusted_errors_pc_rate
        sum_waer = 0.0  # wrong_adjusted_errors_pc_rate

        total_time = 0.0

        # Número de funções de teste
        n_tests = len(test_functions)

        for (test_func, path_str) in test_functions:
            start_time = time.perf_counter()
            # Chama a função de teste, que retorna dicionário com a chave "ocpc" e seus valores
            results_dict = test_func(path_str, k_max, alfa, lamda, f)
            end_time = time.perf_counter()

            elapsed = end_time - start_time
            total_time += elapsed

            # Extrair somente a parte “ocpc” do dicionário retornado
            ocpc = results_dict.get("ocpc", {})

            # Acumular cada métrica
            sum_cdor += ocpc.get("correct_detected_outliers_rate", 0.0)
            sum_wdor += ocpc.get("wrong_detected_outliers_rate", 0.0)
            sum_caer += ocpc.get("correct_adjusted_errors_pc_rate", 0.0)
            sum_waer += ocpc.get("wrong_adjusted_errors_pc_rate", 0.0)

        # Médias (para cada métrica)
        avg_cdor = sum_cdor / n_tests
        avg_wdor = sum_wdor / n_tests
        avg_caer = sum_caer / n_tests
        avg_waer = sum_waer / n_tests

        # Objetivos construídos a partir dessas médias:
        # 1) | avg_cdor – 0.1 |      (quanto mais próximo de 0.1, melhor)
        # 2) avg_wdor              (quanto menor, melhor)
        # 3) | avg_caer – 0.1 |     (quanto mais próximo de 0.1, melhor)
        # 4) avg_waer              (quanto menor, melhor)
        # 5) total_time            (quanto menor, melhor)
        obj1 = abs(avg_cdor - 0.1)
        obj2 = avg_wdor
        obj3 = abs(avg_caer - 0.1)
        obj4 = avg_waer
        obj5 = total_time

        return (obj1, obj2, obj3, obj4, obj5)

    except Exception as e:
        # Penaliza indivíduos que causarem erro: fitness altos
        print(f"Erro na avaliação do indivíduo {individual}: {e}")
        # Retorna valores muito grandes para cada objetivo
        return (1.0, 1.0, 1.0, 1.0, 1e3)

# -----------------------------------------------------------------------------
# Função de mutação (igual à sua, com restrição de limites)
def mutate_with_limits(individual, mu, sigma, indpb):
    mutated_ind = tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb)[0]
    # Limitar cada parâmetro após a mutação
    mutated_ind[0] = max(2,  min(10, int(round(mutated_ind[0]))))   # k_max
    mutated_ind[1] = max(0.1, min(1.0, mutated_ind[1]))            # alfa
    mutated_ind[2] = max(0.1, min(1.0, mutated_ind[2]))            # lamda
    mutated_ind[3] = max(0.5, min(1.5, mutated_ind[3]))            # f
    return mutated_ind,

# -----------------------------------------------------------------------------
# Registrar as operações genéticas
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_with_limits, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selNSGA2)          # NSGA-II para seleção multiobjetivo
toolbox.register("evaluate", evaluate_individual)   # Nossa nova função de avaliação

# -----------------------------------------------------------------------------
def run_nsga2_parallel(seed=None):
    random.seed(seed)
    np.random.seed(seed)

    pop_size = 100
    generations = 100
    cxpb, mutpb = 0.9, 0.1

    population = toolbox.population(n=pop_size)

    with yaspin(Spinners.dots, text="Rodando NSGA-II...") as sp:
        # O algorithms.eaMuPlusLambda usa a função `map` registrada na toolbox
        # para paralelizar avaliações (via Pool). Ele retorna (população_final, logbook)
        population, logbook = algorithms.eaMuPlusLambda(
            population, toolbox,
            mu=pop_size,            # número de pais
            lambda_=pop_size,       # número de filhos gerados
            cxpb=cxpb,
            mutpb=mutpb,
            ngen=generations,
            stats=None,
            halloffame=None,
            verbose=False
        )
        sp.text = "NSGA-II concluído!"
        sp.ok("✅")

    # Extrair a primeira fronteira de Pareto (não-dominados) da última geração
    pareto_front = tools.sortNondominated(population, k=pop_size, first_front_only=True)[0]
    return pareto_front

# -----------------------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()  # continua presente apenas para Windows, mas não há Pool

    total_runs = 5
    all_pareto_points = []

    # **NÃO USAR Pool nem registrar toolbox.map**
    # Ou seja, não há:
    #   with Pool(...) as pool:
    #       toolbox.register("map", pool.map)
    #       ...
    #
    # Em vez disso, tudo roda de forma sequencial:

    for run_idx in tqdm(range(total_runs), desc="Executando runs do NSGA-II"):
        pareto = run_nsga2_parallel(seed=run_idx)
        for ind in pareto:
            all_pareto_points.append({
                'k_max': int(ind[0]),
                'alfa': float(ind[1]),
                'lamda': float(ind[2]),
                'f': float(ind[3]),
                'obj1_cdor_dist_to_0.1': ind.fitness.values[0],
                'obj2_wdor': ind.fitness.values[1],
                'obj3_caer_dist_to_0.1': ind.fitness.values[2],
                'obj4_waer': ind.fitness.values[3],
                'obj5_time': ind.fitness.values[4]
            })

    # Salvar CSV e plotar, etc. (mesmo de antes)
    df_results_mo = pd.DataFrame(all_pareto_points)
    df_results_mo.to_csv("resultados_otimizacao_ga_multiobjetivo.csv", index=False)
    print("\n✅ Otimização multiobjetivo concluída! (CSV gerado)")

    # Plotar a fronteira de Pareto agregada (erro de detecção × tempo) como exemplo
    if not df_results_mo.empty:
        plt.figure(figsize=(10, 8))
        # Como exemplo, plotaremos: |cdor - 0.1| vs tempo (objetivos 1 e 5)
        plt.scatter(df_results_mo['obj1_cdor_dist_to_0.1'],
                    df_results_mo['obj5_time'],
                    c='blue', alpha=0.6,
                    label='Fronteira de Pareto (dist_cdor × tempo)')
        plt.title("Pareto (|correct_detected_outliers_rate – 0.1| vs tempo)")
        plt.xlabel("|correct_detected_outliers_rate – 0.1| (a minimizar)")
        plt.ylabel("Tempo total (s) (a minimizar)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("fronteira_pareto_multiobjetivo.png")
        plt.show()
    else:
        print("Nenhum resultado para plotar a fronteira de Pareto.")

    # Exemplos: Melhor indivíduo por objetivo isolado (objetivo 1 e 5, só para demonstração)
    if not df_results_mo.empty:
        best_cdor = df_results_mo.loc[df_results_mo['obj1_cdor_dist_to_0.1'].idxmin()]
        print("\nIndivíduo com |cdor – 0.1| menor (objetivo 1):")
        print(best_cdor.to_dict())

        best_time = df_results_mo.loc[df_results_mo['obj5_time'].idxmin()]
        print("\nIndivíduo com menor tempo (objetivo 5):")
        print(best_time.to_dict())
