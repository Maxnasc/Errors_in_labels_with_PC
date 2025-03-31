import numpy as np
import matplotlib.pyplot as plt 
from ocpc_py import MultiClassPC
import pandas as pd
import os
from sklearn.datasets import load_iris
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from cleanlab import Datalab

def get_CL_label_correction(X, Y_error, Y_original):
    """
    Identifica e corrige possíveis erros de rótulos em um conjunto de dados utilizando aprendizado confiante (Confident Learning).
    Parâmetros:
        X (array-like): Conjunto de características (features) utilizado para treinar o modelo.
        Y_error (array-like): Rótulos possivelmente incorretos associados aos dados.
        Y_original (array-like): Rótulos originais (verdadeiros) para comparação.
    Retorna:
        pandas.DataFrame: Um DataFrame contendo informações sobre os rótulos identificados como problemáticos, 
        incluindo os rótulos fornecidos, os rótulos previstos e os rótulos originais.
    Detalhes:
        - Treina um modelo de regressão logística para prever probabilidades de classe com base nos dados fornecidos.
        - Utiliza a biblioteca `Datalab` para identificar problemas nos rótulos com base nas probabilidades previstas.
        - Calcula a porcentagem de rótulos incorretos antes e depois da aplicação do aprendizado confiante.
        - Exibe no console a porcentagem de rótulos corrigidos e se houve melhora, piora ou nenhuma alteração após a correção.
    Exemplo de uso:
        issues = get_CL_label_correction(X, Y_error, Y_original)
    """
    # Treinar modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X, Y_error)
    probs = model.predict_proba(X)

    # Método moderno (recomendado)
    lab = Datalab(data={"y": Y_error}, label_name="y")
    lab.find_issues(features=X, pred_probs=probs)
    issues = lab.get_issues('label')
    issues['original_labels'] = Y_original

    porcentagem_labels_erradas = (issues['given_label'] != issues['original_labels']).mean()
    porcentagem_labels_erradas_depois_do_CL = (issues['predicted_label'] != issues['original_labels']).mean()
    score_correcao = porcentagem_labels_erradas - porcentagem_labels_erradas_depois_do_CL

    print(f'porcentagem_labels_erradas: {round(porcentagem_labels_erradas, 2)}%')
    print(f'porcentagem_labels_erradas_depois_do_CL: {round(porcentagem_labels_erradas_depois_do_CL, 2)}%')
    if score_correcao < 0:
        print(f'Piora de : {round(score_correcao, 2)}%')
    elif score_correcao > 0:
        print(f'Melhora de : {round(score_correcao, 2)}%')
    else:
        print(f'Sem alteração: {round(score_correcao, 2)}%')
        
    # print(f'score_correcao (positivo = melhora; negativo = piora): {score_correcao}')
        
    return issues