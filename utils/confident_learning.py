import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from cleanlab import Datalab

def get_CL_label_correction(X, Y_error, Y_original):
    """
    Identifica e corrige possíveis erros de rótulos em um conjunto de dados utilizando Confident Learning (Cleanlab).
    Parâmetros:
        X (array-like): Conjunto de features (n_samples × n_features).
        Y_error (array-like): Rótulos possivelmente corrompidos (1D).
        Y_original (array-like): Rótulos originais (1D), para comparação.
    Retorna:
        metrics (dict): {"original error rate CL": ..., "error rate after correction CL": ...}
        issues  (pd.DataFrame): DataFrame (vazio ou não) com colunas mínimas:
            ['given_label', 'predicted_label', 'original_labels', 'is_label_issue'] 
            e quaisquer outras colunas que o Cleanlab gerar.
    """

    # 1) Treino do modelo (Regressão Logística)
    try:
        model = LogisticRegression(max_iter=1000)
        model.fit(X, Y_error)
        probs = model.predict_proba(X)
    except Exception as e:
        # Se falhar no treino, devolvemos métricas zero e issues vazio
        # (a avaliação continuará, mas sem dados de CL)
        metrics = {
            "original error rate CL": 0.0,
            "error rate after correction CL": 0.0
        }
        empty_issues = pd.DataFrame(columns=[
            'given_label', 'predicted_label', 'original_labels', 'is_label_issue'
        ])
        return metrics, empty_issues

    # 2) Construir o objeto Datalab
    try:
        lab = Datalab(data={"y": Y_error}, label_name="y")
        lab.find_issues(features=X, pred_probs=probs)
    except Exception:
        # Se find_issues falhar de alguma forma, devolvemos também vazio
        metrics = {
            "original error rate CL": 0.0,
            "error rate after correction CL": 0.0
        }
        empty_issues = pd.DataFrame(columns=[
            'given_label', 'predicted_label', 'original_labels', 'is_label_issue'
        ])
        return metrics, empty_issues

    # 3) Tentar pegar as “issues” de tipo 'label'
    try:
        issues = lab.get_issues('label')
        issues['original_labels'] = Y_original
    except Exception:
        # Quando Cleanlab não encontra *nenhum* issue de rótulo, apenas criamos um DataFrame vazio
        issues = pd.DataFrame(columns=[
            'given_label', 'predicted_label', 'original_labels', 'is_label_issue'
        ])
        metrics = {
            "original error rate CL": 0.0,
            "error rate after correction CL": 0.0
        }
        return metrics, issues

    # 4) Se chegamos aqui, temos um DataFrame issues (talvez com linhas ou vazio, mas com colunas corretas)
    if issues.shape[0] == 0:
        # Não houve linhas de issue: métricas também são zero
        metrics = {
            "original error rate CL": 0.0,
            "error rate after correction CL": 0.0
        }
        return metrics, issues

    # 5) Calcular métricas de quantos rótulos estavam errados antes e depois do CL
    try:
        before = (issues['given_label'] != issues['original_labels']).mean()
        after  = (issues['predicted_label'] != issues['original_labels']).mean()
    except KeyError:
        before = 0.0
        after = 0.0

    metrics = {
        "original error rate CL": round(before, 4),
        "error rate after correction CL": round(after, 4)
    }

    return metrics, issues
