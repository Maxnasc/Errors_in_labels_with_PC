from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from PC_LabelCorrector.PC_LabelCorrector import PC_LabelCorrector
from utils.confident_learning import get_CL_label_correction
from utils.utils import get_dataset_with_error, save_metrics_to_json_file

def plot_scatter(labels: list, X, iris, ax):  # Pass the Axes object
    scatter = ax.scatter(X[:,0], X[:,1], c=labels)
    ax.set(xlabel=iris.feature_names[0], ylabel=iris.feature_names[1])
    _ = ax.legend(
        scatter.legend_elements()[0], iris.target_names, loc="lower right", title="Classes"
    )

def test_load_iris_datset(outlier_detection_OCPC: bool):
    iris = load_iris()
    erro_proposto = 0.1
    results = {}
    iris_with_error = get_dataset_with_error(iris.data, iris.target, erro_proposto)

    labels_wrong_before_adjustments = 0
    for i, y in enumerate(iris_with_error['target']):
        if y != iris.target[i]:
            labels_wrong_before_adjustments += 1

    lc = PC_LabelCorrector(detect_outlier_with_ocpc=outlier_detection_OCPC)
    Y_adjusted = lc.run(X=iris_with_error["data"], Y=iris_with_error["target"])
    
    # Comparação com o CL
    CL_issues = get_CL_label_correction(iris_with_error["data"], iris_with_error["target"], iris.target)
    
    metrics = {"original error rate PC_LabelCorrection": lc.metrics['original error rate']} | {"error rate after correction PC_LabelCorrection": lc.metrics['error rate after correction']} | CL_issues
    
    path='tests/load_iris/comparation'
    save_metrics_to_json_file(path=path, metrics=metrics)
    
    # Create a figure and two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))  # 2 rows, 1 column

    # Plot on each subplot
    plt.subplot(2,1,1)
    plot_scatter(labels=iris.target, X=iris.data, iris=iris, ax=ax1)
    ax1.set_title('Original Labels')  # Set title for the first subplot
    
    plt.subplot(2,1,2)
    plot_scatter(labels=Y_adjusted, X=iris.data, iris=iris, ax=ax2)
    ax2.set_title('Adjusted Labels')  # Set title for the second subplot
    
    plt.tight_layout() # Adjust layout to prevent overlapping
    plt.savefig('figura_teste.png')  # Save the whole figure

    
    return metrics
    
if __name__ == "__main__":
    test_load_iris_datset()