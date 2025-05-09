import json
import numpy as np
import pandas as pd
from copy import deepcopy

def get_dataset_with_error(X, Y_original, erro_proposto):

    def alterar_rotulos(Y, percentual, random_state=None):
        """
        Alters the labels of Y by a given percentage.

        Args:
            Y: Original labels
            percentual: Percentage of labels to alter
            random_state: Seed for reproducibility

        Returns:
            Altered labels
        """
        np.random.seed(random_state)  # For reproducibility
        Y_altered = Y.copy()
        classes = np.unique(Y)

        for classe in classes:
            class_indices = np.where(Y == classe)[0]  # Get indices of the class
            n_to_alter = int(len(class_indices) * percentual)
            chosen_indices = np.random.choice(
                class_indices, n_to_alter, replace=False
            )

            # Choose new random labels, different from the original
            for idx in chosen_indices:
                new_classes = np.setdiff1d(classes, Y[idx])  # Avoid the same label
                Y_altered[idx] = np.random.choice(new_classes)

        return Y_altered

    Y = alterar_rotulos(Y_original, erro_proposto)

    # Rebuild data_with_error
    data_with_error = {"data": X, "target": Y, "Y_original": Y_original}

    return data_with_error

def save_metrics_to_json_file(path: str, metrics: dict):
    
        # Save results to a JSON file
        if ".json" not in path:
            path = path + ".json"

        with open(path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Results saved to {path}")
        
def save_metrics_to_csv_file(path: str, metrics: dict):

        # Coverter os valores dentro de metrics para listas
        new_metrics = deepcopy(metrics)
        for key, value in new_metrics.items():
            list_val = [value]
            new_metrics[key] = list_val
            
        df = pd.DataFrame(new_metrics)
        
        if ".csv" not in path:
            path = path + ".csv"
        
        df.to_csv(path);

        print(f"Results saved to {path}")