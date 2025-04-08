import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor

def generate_data():
    """Gera um conjunto de dados com inliers e outliers."""
    rng = np.random.RandomState(42)
    X_inliers = 0.3 * rng.randn(100, 2)  # 100 pontos normais
    X_outliers = rng.uniform(low=-4, high=4, size=(10, 2))  # 10 outliers
    return np.vstack((X_inliers, X_outliers))

def detect_outliers_lof(X, n_neighbors=50, contamination='auto'):
    """Aplica o Local Outlier Factor (LOF) para detectar outliers."""
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    y_pred = lof.fit_predict(X)
    scores = lof.negative_outlier_factor_
    return y_pred, scores

def plot_results(X, y_pred):
    """Plota os resultados da detecção de outliers."""
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', edgecolors='k')
    
    # Destacando os outliers com um círculo vermelho
    outlier_mask = y_pred == -1
    plt.scatter(X[outlier_mask, 0], X[outlier_mask, 1], edgecolors='r', facecolors='none', s=100, label="Outliers")
    
    plt.legend()
    plt.title("Detecção de Outliers com LOF")
    plt.show()

def main():
    X = generate_data()
    y_pred, _ = detect_outliers_lof(X, n_neighbors=int(len(X)/2), contamination='auto')
    plot_results(X, y_pred)
    
    # Filtrando os inliers
    X_clean = X[y_pred == 1]
    print(f"Número de inliers: {len(X_clean)}")
    print(f"Número de outliers: {len(X) - len(X_clean)}")

if __name__ == "__main__":
    main()
