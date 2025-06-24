import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import os
import time
from yaspin import yaspin
from yaspin.spinners import Spinners
import multiprocessing # Importação adicionada para paralelização

# Importar as quatro funções de teste (presume-se que existam nos caminhos especificados)
from tests.breast_cancer.breast_cancer import test_breast_cancer_dataset
from tests.load_iris.load_iris import test_load_iris_dataset
from tests.load_wine.load_wine import test_load_wine_dataset
from tests.sintetic_2D_dataset.sintetic_2D_dataset import test_2D_sintetic_dataset

# -------------------------------------------------------------------
# 1. Criar as classes de Fitness e Individual para MULTI-OBJETIVOS (3 objetivos)
#    avg_cdor será MAXIMIZADO (peso positivo: 1.0)
#    avg_caer e total_time serão MINIMIZADOS (pesos negativos: -1.0, -1.0)
if "Fitness3Obj" not in creator.__dict__:
    creator.create("Fitness3Obj", base.Fitness, weights=(1.0, -1.0, -1.0))

if "Individual" not in creator.__dict__:
    creator.create("Individual", list, fitness=creator.Fitness3Obj)

# -------------------------------------------------------------------
# 2. Espaço de busca (quatro hiperparâmetros do PC_LabelCorrector)
toolbox = base.Toolbox()
# k_max é um inteiro entre 2 e 10
toolbox.register("attr_k_max", random.randint, 2, 10)
# alfa é um float entre 0.1 e 1.0
toolbox.register("attr_alfa", random.uniform, 0.1, 1.0)
# lamda é um float entre 0.1 e 1.0
toolbox.register("attr_lamda", random.uniform, 0.1, 1.0)
# f é um float entre 0.5 e 1.5
toolbox.register("attr_f", random.uniform, 0.5, 1.5)

# Definindo como um indivíduo é inicializado, usando os atributos definidos
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_k_max,
                  toolbox.attr_alfa,
                  toolbox.attr_lamda,
                  toolbox.attr_f),
                 n=1)
# Definindo como uma população é inicializada
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
    e montamos um vetor-fitness:
        1) avg_cdor (obj1) - a ser maximizado
        2) avg_caer (obj3) - a ser minimizado
        3) total_time (obj5) - a ser minimizado

    Se ocorrer qualquer exceção (por exemplo, “Lengths must match to compare”),
    penaliza o indivíduo com valores que o tornam indesejável para os objetivos
    (valor baixo para o que deve ser maximizado e valores altos para o que deve ser minimizado).
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
            # Passa os parâmetros individuais para a função de teste
            results_dict = test_func(path_str, k_max, alfa, lamda, f)
            end_time = time.perf_counter()
            elapsed = end_time - start_time
            total_time += elapsed # Acumula o tempo total de execução

            ocpc = results_dict.get("ocpc", {})
            # Supondo que “erros_de_rotulo_ajustados_corretamente” e “novos_erros_gerados”
            # sempre existam como float ou int. Se faltar, retorna 0.0 por default.
            sum_cdor += float(ocpc.get("erros_de_rotulo_ajustados_corretamente", 0.0))
            sum_caer += float(ocpc.get("novos_erros_gerados", 0.0))

        # Calcula as médias dos erros e o tempo total
        avg_cdor = sum_cdor / n_tests
        avg_caer = sum_caer / n_tests

        # Retorna a tupla de valores de fitness
        return (avg_cdor, avg_caer, total_time)

    except Exception as e:
        # Em caso de falha, imprime um log de erro e penaliza o indivíduo com valores que
        # o tornam indesejável para os objetivos.
        # Para maximizar obj1, retorna um valor baixo (e.g., 0.0).
        # Para minimizar obj3 e obj5, retorna valores altos (e.g., 1.0, 1e3).
        print(f"[ERROR] Avaliação falhou para indivíduo {individual}: {e}")
        return (0.0, 1.0, 1e3) # Valores para penalização: 0.0 para maximização, altos para minimização

# -------------------------------------------------------------------
# 5. Função de mutação (com limites nos parâmetros)
def mutate_with_limits(individual, mu, sigma, indpb):
    # Aplica mutação gaussiana ao indivíduo
    mutated_ind = tools.mutGaussian(individual, mu=mu, sigma=sigma, indpb=indpb)[0]
    # Garante que os valores mutados estejam dentro dos limites definidos
    mutated_ind[0] = max(2,  min(10, int(round(mutated_ind[0]))))   # k_max deve ser inteiro
    mutated_ind[1] = max(0.1, min(1.0, mutated_ind[1]))             # alfa
    mutated_ind[2] = max(0.1, min(1.0, mutated_ind[2]))             # lamda
    mutated_ind[3] = max(0.5, min(1.5, mutated_ind[3]))             # f
    return mutated_ind,

# -------------------------------------------------------------------
# 6. Registrar as operações genéticas no toolbox
toolbox.register("mate",    tools.cxTwoPoint) # Operador de cruzamento de dois pontos
toolbox.register("mutate",  mutate_with_limits, mu=0, sigma=0.1, indpb=0.2) # Operador de mutação com limites
toolbox.register("select",  tools.selNSGA2)          # Seleção NSGA-II
toolbox.register("evaluate", evaluate_individual)    # Função de avaliação

# -------------------------------------------------------------------
def run_nsga2(seed=None):
    """
    Executa o NSGA-II com paralelismo.
    Retorna um objeto ParetoFront() contendo todos os indivíduos não-dominados.
    """
    random.seed(seed)
    np.random.seed(seed)

    pop_size = 40
    generations = 20
    cxpb, mutpb = 0.9, 0.1

    # 1) Criar população inicial
    population = toolbox.population(n=pop_size)

    # 2) Container para não-dominados ao longo das gerações
    pareto_hof = tools.ParetoFront()

    # 3) Rodar o NSGA-II utilizando multiprocessing.Pool para avaliação paralela
    # Criar um Pool de processos. O número de processos padrão é o número de núcleos da CPU.
    with multiprocessing.Pool() as pool:
        # Atribuir o pool.map à função map do toolbox
        toolbox.register("map", pool.map)
        with yaspin(Spinners.dots, text="Rodando NSGA-II (paralelo)...") as spinner:
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

    print(f"[INFO] {len(pareto_hof)} indivíduos na fronteira de Pareto final.")
    return pareto_hof

# -------------------------------------------------------------------
if __name__ == "__main__":
    # Garante compatibilidade no Windows para o uso de multiprocessing
    multiprocessing.freeze_support()

    # 1. Executar NSGA-II
    pareto_front = run_nsga2(seed=42)

    # 2. Extrair todos os indivíduos não-dominados e montar DataFrame
    all_pareto_points = []
    for ind in pareto_front:
        all_pareto_points.append({
            'k_max':        int(ind[0]),
            'alfa':         float(ind[1]),
            'lamda':        float(ind[2]),
            'f':            float(ind[3]),
            'obj1_cdor':    ind.fitness.values[0],
            'obj3_caer':    ind.fitness.values[1],
            'obj5_time':    ind.fitness.values[2]
        })

    df_results = pd.DataFrame(all_pareto_points)
    os.makedirs("hiperParametrization", exist_ok=True)

    # 3. Salvar CSV com a fronteira de Pareto
    df_results.to_csv("resultados_pareto_3objetivos.csv", index=False)
    print("✅ Fronteira de Pareto salva em 'resultados_pareto_3objetivos.csv'.")

    # 4. Calcular soma ponderada (pesos: obj1→1.5, obj3→1.2, obj5→1.0)
    # A soma ponderada é usada para encontrar um "melhor" indivíduo dentro da fronteira de Pareto,
    # caso haja necessidade de uma única solução de compromisso.
    # Note que aqui os pesos são para a soma ponderada de "melhor indivíduo" e não diretamente
    # para a otimização multi-objetivo do NSGA-II.
    # Para consistência com os objetivos de MAXIMIZAÇÃO (cdor) e MINIMIZAÇÃO (caer, tempo),
    # a soma ponderada é calculada para encontrar o "melhor" ponto, onde um menor valor total
    # indica um melhor compromisso.
    # Para 'obj1_cdor' que está sendo maximizado no GA, multiplicamos por um peso negativo
    # na soma ponderada se quisermos minimizar a "soma ponderada geral".
    # No entanto, se a intenção é que um valor mais alto de cdor seja "melhor" na soma ponderada,
    # mantemos o peso positivo. Pela sua solicitação inicial, "aumentar os valores de gerações"
    # e "melhorar as métricas desejadas", e o cálculo da soma ponderada para o "menor weighted_sum",
    # para 'obj1_cdor' (que o GA maximiza), um peso positivo na soma ponderada significa
    # que um 'cdor' maior contribuirá para uma 'weighted_sum' maior, o que vai contra
    # a ideia de buscar o 'idxmin'.
    # ASSUMindo que 'weighted_sum' deve ser MINIMIZADA e um 'cdor' alto é bom,
    # o peso para 'obj1_cdor' na 'weighted_sum' deve ser NEGATIVO.
    # Se a intenção é que 'weighted_sum' seja uma pontuação onde 'MENOR' é 'MELHOR',
    # e 'avg_cdor' (obj1) deve ser 'MAIOR' para ser 'MELHOR', então:
    # 1.5 * (-df_results['obj1_cdor']) + 1.2 * df_results['obj3_caer'] + 1.0 * df_results['obj5_time']
    # Contudo, para manter a lógica do seu código original de soma ponderada (que minimiza a soma),
    # e para que o obj1_cdor alto resulte em uma melhor pontuação na soma ponderada,
    # e como o objetivo é buscar o idxmin(), precisamos inverter o sinal para obj1_cdor na soma ponderada
    # se o objetivo do GA é maximizá-lo e queremos que ele contribua para uma "melhor" (menor) soma ponderada.
    # Se mantivermos 1.5 * df_results['obj1_cdor'], e obj1_cdor é maximizado pelo GA,
    # então o idxmin() de weighted_sum favorecerá um obj1_cdor baixo, o que é o oposto do que se quer.
    # A solução para que o "melhor indivíduo" da soma ponderada reflita seus objetivos (maximizar cdor, minimizar caer/tempo)
    # E para que `idxmin` funcione corretamente, é fazer com que a contribuição de `obj1_cdor` seja negativa
    # na `weighted_sum`.
    df_results['weighted_sum'] = (
        -1.5 * df_results['obj1_cdor'] + # Negativo para que maximizar cdor leve a uma menor soma ponderada
        1.2 * df_results['obj3_caer'] +
        1.0 * df_results['obj5_time']
    )


    # 5. Identificar melhor indivíduo global (menor weighted_sum)
    if not df_results.empty:
        idx_best = df_results['weighted_sum'].idxmin()
        best_row = df_results.loc[idx_best]

        # 6. Salvar melhor indivíduo em CSV separado
        df_best = best_row.to_frame().T
        df_best.to_csv("hiperParametrization/best-subject.csv", index=False)
        print("✅ Melhor indivíduo salvo em 'hiperParametrization/best-subject.csv'.")

        # 7. (Opcional) Plotar Pareto (obj1 × obj5)
        plt.figure(figsize=(8, 6))
        # No plot, obj1_cdor ainda é apresentado como é (para visualização da frente de Pareto real)
        plt.scatter(df_results['obj1_cdor'],
                    df_results['obj5_time'],
                    alpha=0.6,
                    label='Fronteira de Pareto (obj1 × obj5)')
        plt.title("Pareto (avg_cdor vs Tempo total)")
        plt.xlabel("avg_cdor (a maximizar)") # Rótulo ajustado para refletir o objetivo
        plt.ylabel("Tempo total (s) (a minimizar)")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("hiperParametrization/pareto_3objetivos.png") # Salva o plot no diretório correto
        plt.show()
    else:
        print("⚠️ Não há resultados de Pareto para salvar ou plotar.")