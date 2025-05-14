import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from deap import base, creator, tools, algorithms
# from multiprocessing import Pool, cpu_count  # Removido paralelismo
from utils.utils import get_dataset_with_error
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
import os
from time import sleep

# Criar tipos apenas se ainda não foram criados
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Espaço de busca
toolbox.register("attr_k_max", random.randint, 2, 20)
toolbox.register("attr_alfa", random.uniform, 0.1, 1.0)
toolbox.register("attr_lamda", random.uniform, 0.1, 1.0)
toolbox.register("attr_buffer", random.choice, [1000])
toolbox.register("attr_f", random.uniform, 0.5, 3.0)
toolbox.register("attr_outlier_rate", random.choice, [0.1])

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_k_max,
                  toolbox.attr_alfa,
                  toolbox.attr_lamda,
                  toolbox.attr_buffer,
                  toolbox.attr_f,
                  toolbox.attr_outlier_rate),
                 n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Carregar datasets e aplicar erro
erro_proposto = 0.1
datasets = {
    "iris": get_dataset_with_error(load_iris().data, load_iris().target, erro_proposto),
    "breast_cancer": get_dataset_with_error(load_breast_cancer().data, load_breast_cancer().target, erro_proposto),
    "wine": get_dataset_with_error(load_wine().data, load_wine().target, erro_proposto)
}

# Função de avaliação para 1 indivíduo
def evaluate_individual(individual):
    k_max, alfa, lamda, buffer, f, outlier_rate = individual
    ocpc_params = {
        'k_max': int(k_max),
        'alfa': float(alfa),
        'lamda': float(lamda),
        'buffer': int(buffer),
        'f': float(f),
        'outlier_rate': float(outlier_rate)
    }

    total_error = 0.0
    try:
        for data in datasets.values():
            lc = PC_LabelCorrector(
                detect_outlier_with_ocpc=True,
                k_max=ocpc_params['k_max'],
                alfa=ocpc_params['alfa'],
                lamda=ocpc_params['lamda'],
                buffer=ocpc_params['buffer'],
                f=ocpc_params['f'],
                outlier_rate=ocpc_params['outlier_rate']
            )
            lc.run(X=data["data"], Y=data["target"])
            total_error += lc.metrics['error rate after correction']
        avg_error = total_error / len(datasets)
        return (avg_error,)
    except Exception as e:
        print(f"Erro ao avaliar indivíduo: {e}")
        return (1.0,)

toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# Função para executar uma rodada do GA sem paralelismo
def run_ga(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    pop_size = 50
    generations = 30
    cxpb, mutpb = 0.7, 0.2

    population = toolbox.population(n=pop_size)

    for gen in tqdm(range(generations), desc="Geração", leave=False):
        offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)

        # Avaliação sequencial (sem paralelismo)
        fitnesses = list(map(evaluate_individual, offspring))

        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        elite = tools.selBest(population, k=1)
        population = toolbox.select(offspring, k=pop_size - 1) + elite

    best = tools.selBest(population, k=1)[0]
    return best, best.fitness.values[0]

def clear_terminal():
    os.system('cls' if os.name == 'nt' else 'clear')

if __name__ == "__main__":
    all_best_individuals = []
    all_best_errors = []

    total_runs = 30
    pbar = tqdm(range(total_runs), desc="🔁 Executando Algoritmo Genético", position=0)

    for i in pbar:
        pbar.set_description(f"🔁 Rodada {i+1}/{total_runs}")
        best_ind, best_error = run_ga(seed=i)
        all_best_individuals.append(best_ind)
        all_best_errors.append(best_error)
        pbar.set_postfix({"Erro Médio": f"{best_error:.4f}"})
        sleep(0.1)

    df_results = pd.DataFrame([{
        'k_max': ind[0],
        'alfa': ind[1],
        'lamda': ind[2],
        'buffer': ind[3],
        'f': ind[4],
        'outlier_rate': ind[5],
        'avg_error_rate': error
    } for ind, error in zip(all_best_individuals, all_best_errors)])

    df_results.to_csv("resultados_otimizacao_ga.csv", index=False)
    print("\n📁 Resultados salvos em 'resultados_otimizacao_ga.csv'")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, 31), all_best_errors, marker='o', linestyle='-', color='blue')
    plt.title("Erro Médio do Melhor Indivíduo por Execução (3 datasets)")
    plt.xlabel("Execução")
    plt.ylabel("Erro Médio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_erros.png")
    plt.show()

    mean_error = np.mean(all_best_errors)
    std_error = np.std(all_best_errors)

    print(f"\n📊 Estatísticas das 30 execuções:")
    print(f"Média da taxa de erro média: {mean_error:.4f}")
    print(f"Desvio padrão: {std_error:.4f}")
