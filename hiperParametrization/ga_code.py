from deap import base, creator, tools, algorithms
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from sklearn.datasets import load_iris
import random
import numpy as np
from tqdm import tqdm  # Import the tqdm library

from utils.utils import get_dataset_with_error


# Definir o tipo de fitness (minimizar a taxa de erro)
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()

# Atributos (parâmetros)
toolbox.register("attr_k_max", random.randint, 2, 20)
toolbox.register("attr_alfa", random.uniform, 0.0, 1.0)
toolbox.register("attr_lamda", random.uniform, 0.0, 1.0)
toolbox.register("attr_buffer", random.choice, [1000])  # Mantive como lista, mas pode ajustar
toolbox.register("attr_f", random.uniform, 0.5, 3.0)
toolbox.register("attr_outlier_rate", random.choice, [0.1]) # Changed to 0.1 for a valid default

# Criar o indivíduo
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_k_max,
                  toolbox.attr_alfa,
                  toolbox.attr_lamda,
                  toolbox.attr_buffer,
                  toolbox.attr_f,
                  toolbox.attr_outlier_rate),
                 n=1)

# Criar a população
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

data = load_iris()
erro_proposto = 0.1
data_with_error = get_dataset_with_error(data.data, data.target, erro_proposto)

# Função de avaliação
def evaluate(individual):
    k_max, alfa, lamda, buffer, f, outlier_rate = individual

    ocpc_params = {
        'k_max': int(k_max),
        'alfa': float(alfa),  # Ensure float conversion
        'lamda': float(lamda), # Ensure float conversion
        'buffer': int(buffer),
        'f': float(f),        # Ensure float conversion
        'outlier_rate': float(outlier_rate) # Ensure float conversion
    }

    try:
        lc = PC_LabelCorrector(detect_outlier_with_ocpc=True,
                              k_max=ocpc_params.get('k_max'),
                              alfa=ocpc_params.get('alfa'),
                              lamda=ocpc_params.get('lamda'),
                              buffer=ocpc_params.get('buffer'),
                              f=ocpc_params.get('f'),
                              outlier_rate=ocpc_params.get('outlier_rate'))
        Y_adjusted = lc.run(X=data_with_error["data"], Y=data_with_error["target"])
        error_rate = lc.metrics['error rate after correction']
        return (error_rate,)

    except Exception as e:
        print(f"Error during evaluation: {e}")
        return (1.0,)  # Return a high error (bad fitness) on failure

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1) # Changed to mutGaussian
toolbox.register("select", tools.selTournament, tournsize=3)

# Configurar o algoritmo genético
population = toolbox.population(n=50)
ngen = 30
cxpb, mutpb = 0.7, 0.2

# Executar o algoritmo com barra de progresso
print("Executando o Algoritmo Genético:")
for gen in tqdm(range(ngen), desc="Gerações"):
    offspring = algorithms.varAnd(population, toolbox, cxpb=cxpb, mutpb=mutpb)
    fitnesses = list(map(toolbox.evaluate, offspring))
    for ind, fit in zip(offspring, fitnesses):
        ind.fitness.values = fit
    population = toolbox.select(offspring, k=len(population))

best_ind = tools.selBest(population, k=1)[0]
print("Melhor indivíduo encontrado:", best_ind)