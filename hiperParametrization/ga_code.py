import itertools
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
from deap import base, creator, tools, algorithms
from multiprocessing import Pool, cpu_count, freeze_support
from utils.utils import get_dataset_with_error
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
import os
from yaspin import yaspin
from yaspin.spinners import Spinners
import time

# Configuração inicial
if "FitnessMin" not in creator.__dict__:
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.FitnessMin)

# Espaço de busca otimizado
toolbox = base.Toolbox()
toolbox.register("attr_k_max", random.randint, 2, 10)
toolbox.register("attr_alfa", random.uniform, 0.1, 1.0)
toolbox.register("attr_lamda", random.uniform, 0.1, 1.0)
toolbox.register("attr_buffer", random.choice, [1000])
toolbox.register("attr_f", random.uniform, 0.5, 1.0)
toolbox.register("attr_outlier_rate", random.choice, [0.1])

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_k_max, toolbox.attr_alfa, toolbox.attr_lamda,
                  toolbox.attr_buffer, toolbox.attr_f, toolbox.attr_outlier_rate),
                 n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Carregar datasets uma única vez
erro_proposto = 0.1
datasets = {
    "iris": get_dataset_with_error(load_iris().data, load_iris().target, erro_proposto),
    "breast_cancer": get_dataset_with_error(load_breast_cancer().data, load_breast_cancer().target, erro_proposto),
    "wine": get_dataset_with_error(load_wine().data, load_wine().target, erro_proposto)
}

# Função de avaliação otimizada
def evaluate_individual(individual):
    try:
        params = {
            'k_max': int(individual[0]),
            'alfa': float(individual[1]),
            'lamda': float(individual[2]),
            'buffer': int(individual[3]),
            'f': float(individual[4]),
            'outlier_rate': float(individual[5]),
            'detect_outlier_with_ocpc': True
        }

        total_error = 0.0
        lc = PC_LabelCorrector(**params)

        for data in datasets.values():
            lc.run(X=data["data"], Y=data["target"])
            total_error += lc.metrics['error rate after correction']

        return (total_error / len(datasets),)
    except Exception:
        return (1.0,)
    
def mutate_with_limits(individual, mu, sigma, indpb):
    mutated_individual = tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb)[0]
    # Aplicar limites para k_max
    mutated_individual[0] = max(2, min(10, int(round(mutated_individual[0])))) # Arredonda para inteiro
    # Aplicar limites para alfa
    mutated_individual[1] = max(0.1, min(1.0, mutated_individual[1]))
    # Aplicar limites para lamda
    mutated_individual[2] = max(0.1, min(1.0, mutated_individual[2]))
    # 'buffer' é uma escolha, então não precisa de limite
    # Aplicar limites para f
    mutated_individual[4] = max(0.5, min(1.0, mutated_individual[4]))
    # 'outlier_rate' é uma escolha, então não precisa de limite
    return mutated_individual,

# Configuração do GA
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", mutate_with_limits, mu=0, sigma=0.1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_ga_parallel(seed=None):
    random.seed(seed)
    np.random.seed(seed)

    pop_size = 50
    generations = 30
    cxpb, mutpb = 0.7, 0.2
    population = toolbox.population(n=pop_size)

    with yaspin(Spinners.dots, text="Rodando gerações da GA...") as sp:
        for gen in range(generations):
            sp.text = f"Rodando geração {gen+1}/{generations} da GA..."
            offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
            fitnesses = list(map(evaluate_individual, offspring))

            for ind, fit in zip(offspring, fitnesses):
                ind.fitness.values = fit

            elite = tools.selBest(population, k=1)
            population = toolbox.select(offspring, k=pop_size - 1) + elite
            time.sleep(0.1)  # Simula um tempo de processamento por geração

        sp.text = "Gerações da GA concluídas!"
        sp.ok("✅")

    best = tools.selBest(population, k=1)[0]
    return best, best.fitness.values[0]

if __name__ == "__main__":
    freeze_support()

    total_runs = 1
    results = []

    with Pool(processes=max(1, cpu_count()-1)) as pool:
            results = list(pool.imap(run_ga_parallel, range(total_runs)))

    all_best_individuals, all_best_errors = zip(*results)

    # Salvar resultados
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

    # Plot e estatísticas (mantido igual)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, total_runs + 1), all_best_errors, marker='o', linestyle='-', color='blue')
    plt.title("Erro Médio do Melhor Indivíduo por Execução")
    plt.xlabel("Execução")
    plt.ylabel("Erro Médio")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_erros.png")
    plt.close()  # Evita mostrar o plot em ambientes headless

    mean_error = np.mean(all_best_errors)
    std_error = np.std(all_best_errors)

    print("\n📊 Estatísticas das execuções:")
    print(f"Média da taxa de erro média: {mean_error:.4f}")
    print(f"Desvio padrão: {std_error:.4f}")