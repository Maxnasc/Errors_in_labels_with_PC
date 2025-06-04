import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, cpu_count, freeze_support
import os
from yaspin import yaspin
from yaspin.spinners import Spinners
import time

# Importar as quatro funções de teste
from tests.breast_cancer.breast_cancer import test_breast_cancer_dataset
from tests.load_iris.load_iris import test_load_iris_dataset
from tests.load_wine.load_wine import test_load_wine_dataset
from tests.sintetic_2D_dataset.sintetic_2D_dataset import test_2D_sintetic_dataset

# -------------------------------------------------------------------
# 1. Criar as classes de Fitness e Individual para MULTI-OBJETIVOS (3 objetivos agora)
#    Todos os três objetivos serão minimizados → pesos = -1.0 cada.
if "Fitness3Obj" not in creator.__dict__:
    creator.create("Fitness3Obj", base.Fitness, weights=(-1.0, -1.0, -1.0))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.Fitness3Obj)

# -------------------------------------------------------------------
# 2. Espaço de busca (quatro hiperparâmetros do PC_LabelCorrector)
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

# -------------------------------------------------------------------
# 3. Lista das quatro funções de teste e seus “paths” correspondentes:
test_functions = [
    (test_breast_cancer_dataset, "breast_cancer"),
    (test_load_iris_dataset,     "load_iris"),
    (test_load_wine_dataset,     "load_wine"),
    (test_2D_sintetic_dataset,   "sintetic_2D_dataset")
]

# -------------------------------------------------------------------
# 4. Função de avaliação adaptada para 3 objetivos
def evaluate_individual(individual):
    """
    Cada indivíduo é [k_max, alfa, lamda, f].
    Chamamos as quatro funções de teste, extraímos as métricas ocpc[…]
    e montamos um vetor-fitness de 3 valores:
        1) |avg_cdor – 0.1|     (obj1)
        2) |avg_caer – 0.1|    (obj3)
        3) total_time           (obj5)
    """
    try:
        k_max = int(individual[0])
        alfa  = float(individual[1])
        lamda = float(individual[2])
        f     = float(individual[3])

        sum_cdor = 0.0
        sum_caer = 0.0
        total_time = 0.0

        n_tests = len(test_functions)

        for (test_func, path_str) in test_functions:
            start_time = time.perf_counter()
            results_dict = test_func(path_str, k_max, alfa, lamda, f)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            total_time += elapsed

            ocpc = results_dict.get("ocpc", {})
            sum_cdor += ocpc.get("correct_detected_outliers_rate", 0.0)
            sum_caer += ocpc.get("correct_adjusted_errors_pc_rate", 0.0)

        avg_cdor = sum_cdor / n_tests
        avg_caer = sum_caer / n_tests

        obj1 = abs(avg_cdor - 0.1)
        obj3 = abs(avg_caer - 0.1)
        obj5 = total_time

        return (obj1, obj3, obj5)

    except Exception as e:
        # Se falhar, penaliza com valores grandes
        print(f"[ERROR] Avaliação falhou para indivíduo {individual}: {e}")
        return (1.0, 1.0, 1e3)

# -------------------------------------------------------------------
# 5. Função de mutação (com limites nos parâmetros)
def mutate_with_limits(individual, mu, sigma, indpb):
    mutated_ind = tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb)[0]
    mutated_ind[0] = max(2,  min(10, int(round(mutated_ind[0]))))   # k_max
    mutated_ind[1] = max(0.1, min(1.0, mutated_ind[1]))             # alfa
    mutated_ind[2] = max(0.1, min(1.0, mutated_ind[2]))             # lamda
    mutated_ind[3] = max(0.5, min(1.5, mutated_ind[3]))             # f
    return mutated_ind,

# -------------------------------------------------------------------
# 6. Registrar as operações genéticas no toolbox
toolbox.register("mate",    tools.cxTwoPoint)
toolbox.register("mutate",  mutate_with_limits, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select",  tools.selNSGA2)          # Seleção NSGA-II
toolbox.register("evaluate", evaluate_individual)    # Função de avaliação

# -------------------------------------------------------------------
def run_nsga2_parallel(seed=None):
    """
    Executa o NSGA-II UMA única vez, paralelizando APENAS a avaliação de indivíduos.
    Retorna um objeto ParetoFront() contendo todos os indivíduos não-dominados.
    """
    random.seed(seed)
    np.random.seed(seed)

    pop_size = 100
    generations = 100
    cxpb, mutpb = 0.9, 0.1

    # 1) Criar população inicial
    population = toolbox.population(n=pop_size)

    # 2) Container para não-dominados ao longo das gerações
    pareto_hof = tools.ParetoFront()

    # 3) Abrir Pool de processos para paralelizar avaliação
    n_cores = cpu_count()
    print(f"[DEBUG] Inicializando Pool com {n_cores} processos.")
    with Pool(processes=n_cores) as pool:
        toolbox.register("map", pool.map)

        # 4) Rodar o NSGA-II
        with yaspin(Spinners.dots, text="Rodando NSGA-II...") as spinner:
            pop_final, logbook = algorithms.eaMuPlusLambda(
                population, toolbox,
                mu=pop_size,
                lambda_=pop_size,
                cxpb=cxpb,
                mutpb=mutpb,
                ngen=generations,
                stats=None,
                halloffame=pareto_hof,
                verbose=False
            )
            spinner.text = "NSGA-II concluído!"
            spinner.ok("✅")

    print(f"[DEBUG] {len(pareto_hof)} indivíduos na fronteira de Pareto final.")
    return pareto_hof

# -------------------------------------------------------------------
if __name__ == "__main__":
    freeze_support()  # Para compatibilidade no Windows

    # 1. Executar NSGA-II UMA vez
    pareto_front = run_nsga2_parallel(seed=42)

    # 2. Extrair todos os indivíduos não-dominados e montar DataFrame
    all_pareto_points = []
    for ind in pareto_front:
        all_pareto_points.append({
            'k_max': int(ind[0]),
            'alfa': float(ind[1]),
            'lamda': float(ind[2]),
            'f': float(ind[3]),
            'obj1_cdor_dist_to_0.1': ind.fitness.values[0],
            'obj3_caer_dist_to_0.1': ind.fitness.values[1],
            'obj5_time': ind.fitness.values[2]
        })

    df_results = pd.DataFrame(all_pareto_points)
    os.makedirs("hiperParametrization", exist_ok=True)

    # 3. Salvar CSV com a fronteira de Pareto
    df_results.to_csv("resultados_pareto_3objetivos.csv", index=False)
    print("✅ Fronteira de Pareto salva em 'resultados_pareto_3objetivos.csv'.")

    # 4. Calcular soma ponderada (pesos: obj1→1.0, obj3→1.5, obj5→1.2)
    df_results['weighted_sum'] = (
        1.0 * df_results['obj1_cdor_dist_to_0.1'] +
        1.5 * df_results['obj3_caer_dist_to_0.1'] +
        1.2 * df_results['obj5_time']
    )

    # 5. Identificar melhor indivíduo global (menor weighted_sum)
    idx_best = df_results['weighted_sum'].idxmin()
    best_row = df_results.loc[idx_best]

    # 6. Salvar melhor indivíduo em CSV separado
    df_best = best_row.to_frame().T
    df_best.to_csv("hiperParametrization/best-subject.csv", index=False)
    print(f"✅ Melhor indivíduo salvo em 'hiperParametrization/best-subject.csv'.")

    # 7. (Opcional) Plotar Pareto (obj1 × obj5)
    if not df_results.empty:
        plt.figure(figsize=(8, 6))
        plt.scatter(df_results['obj1_cdor_dist_to_0.1'],
                    df_results['obj5_time'],
                    c='blue', alpha=0.6,
                    label='Fronteira de Pareto (obj1 × obj5)')
        plt.title("Pareto (|cdor–0.1| vs Tempo total)")
        plt.xlabel("|correct_detected_outliers_rate – 0.1| (a minimizar)")
        plt.ylabel("Tempo total (s) (a minimizar)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("pareto_3objetivos.png")
        plt.show()
    else:
        print("Nenhum resultado para plotar.")
