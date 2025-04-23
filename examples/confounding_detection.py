"""
Example script demonstrating how to use the conditional directed information metric
to detect confounding in time series data.
"""

import numpy as np
import matplotlib.pyplot as plt
from interfere.causal_information import ConditionalDirectedInformationMetric
from interfere.interventions import SignalIntervention


def main():
    """
    Main function demonstrating confounding detection.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data with a confounder
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    
    # Z is a confounder that affects both X and Y
    Z = np.sin(2 * t).reshape(-1, 1)
    X = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    Y = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    
    # Create an intervention on X
    intervention = SignalIntervention(0, lambda t: 1.0)  # Fix X to 1.0
    
    # Apply the intervention to get interventional data
    X_int = X.copy()
    X_int[:, 0] = 1.0  # Fix X to 1.0
    Y_int = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    
    # Plot the data
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(t, Z, label='Z (Confounder)')
    plt.title('Confounder (Z)')
    plt.legend()
    
    plt.subplot(3, 1, 2)
    plt.plot(t, X, label='X (Before Intervention)')
    plt.plot(t, X_int, label='X (After Intervention)', linestyle='--')
    plt.title('X Before and After Intervention')
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(t, Y, label='Y (Before Intervention)')
    plt.plot(t, Y_int, label='Y (After Intervention)', linestyle='--')
    plt.title('Y Before and After Intervention')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('images/confounding_example.png')
    plt.close()
    
    # Use the metric to detect confounding
    metric = ConditionalDirectedInformationMetric(threshold=0.05)
    
    # Combine data for the metric
    X_combined = np.hstack([X, Y, Z])
    X_int_combined = np.hstack([X_int, Y_int, Z])
    
    # Create a dummy prediction (not used in confounding detection)
    X_do_pred = X_int_combined.copy()
    
    # Calculate confounding score
    result = metric(X_combined, X_int_combined, X_do_pred, intervention)
    
    # Print the results
    print("Confounding Detection Results:")
    print(f"Confounding score: {result['score']:.4f}")
    
    if result["confounding"]:
        print("\nDetected confounding relationships:")
        for (i, j), score in result["confounding"].items():
            var_names = ["X", "Y", "Z"]
            print(f"{var_names[i]} <-> {var_names[j]}: {score:.4f}")
    else:
        print("\nNo confounding detected.")
    
    # Create a heatmap of confounding relationships
    if result["confounding"]:
        plt.figure(figsize=(8, 6))
        
        # Create a matrix of confounding scores
        n_vars = X_combined.shape[1]
        conf_matrix = np.zeros((n_vars, n_vars))
        
        for (i, j), score in result["confounding"].items():
            conf_matrix[i, j] = score
            conf_matrix[j, i] = score  # Symmetric
        
        # Plot the heatmap
        plt.imshow(conf_matrix, cmap='YlOrRd')
        plt.colorbar(label='Confounding Score')
        
        # Add labels
        var_names = ["X", "Y", "Z"]
        plt.xticks(range(n_vars), var_names)
        plt.yticks(range(n_vars), var_names)
        
        plt.title('Confounding Relationship Heatmap')
        plt.tight_layout()
        plt.savefig('images/confounding_heatmap.png')
        plt.close()


if __name__ == "__main__":
    main() 