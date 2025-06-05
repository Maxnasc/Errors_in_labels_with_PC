import random
import time

from deap import base, creator, tools

# (copie aqui a mesma assinatura de creator e toolbox que você já tem,
#  mas sem usar Pool nem eaMuPlusLambda)

# As suas funções de teste
from tests.breast_cancer.breast_cancer import test_breast_cancer_dataset
from tests.load_iris.load_iris import test_load_iris_dataset
from tests.load_wine.load_wine import test_load_wine_dataset
from tests.sintetic_2D_dataset.sintetic_2D_dataset import test_2D_sintetic_dataset

test_functions = [
    (test_breast_cancer_dataset, "breast_cancer"),
    (test_load_iris_dataset,     "load_iris"),
    (test_load_wine_dataset,     "load_wine"),
    (test_2D_sintetic_dataset,   "sintetic_2D_dataset")
]

# Replique apenas a função de avaliação (sem paralelismo)
def evaluate_individual_sequential(individual):
    k_max = int(individual[0])
    alfa  = float(individual[1])
    lamda = float(individual[2])
    f     = float(individual[3])

    sum_correct_adjusted = 0.0
    sum_wrong_generated  = 0.0
    total_time = 0.0

    n_tests = len(test_functions)

    for (test_func, path_str) in test_functions:
        print(f"\n--- Executando teste '{path_str}' para indivíduo {individual} ---")
        start_time = time.perf_counter()
        results_dict = test_func(path_str, k_max, alfa, lamda, f)
        end_time = time.perf_counter()
        elapsed = end_time - start_time
        print(f"Tempo desse teste: {elapsed:.3f}s")

        ocpc = results_dict.get("ocpc", {})
        corr = ocpc.get("erros_de_rotulo_ajustados_corretamente", None)
        wrong = ocpc.get("novos_erros_gerados", None)

        print(f"ocpc retornou as chaves: {list(ocpc.keys())}")
        print(f" → erros_de_rotulo_ajustados_corretamente = {corr}")
        print(f" → novos_erros_gerados = {wrong}")

        # Use 0.0 como fallback caso não encontre a chave
        sum_correct_adjusted += ocpc.get("erros_de_rotulo_ajustados_corretamente", 0.0)
        sum_wrong_generated  += ocpc.get("novos_erros_gerados", 0.0)
        total_time += elapsed

    avg_correct = sum_correct_adjusted / n_tests
    avg_wrong   = sum_wrong_generated  / n_tests

    print("\n=== Resumo para indivíduo", individual, "===")
    print(f"sum_correct_adjusted = {sum_correct_adjusted:.5f}   → avg_correct = {avg_correct:.5f}")
    print(f"sum_wrong_generated  = {sum_wrong_generated:.5f}   → avg_wrong   = {avg_wrong:.5f}")
    print(f"total_time           = {total_time:.5f}s")

    return (avg_correct, avg_wrong, total_time)


if __name__ == "__main__":
    # Exemplo de indivíduo qualquer (você pode trocar para testar vários)
    indiv = [5, 0.5, 0.5, 1.0]  # k_max=5, alfa=0.5, lamda=0.5, f=1.0
    evaluate_individual_sequential(indiv)
