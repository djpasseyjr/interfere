"""
Evaluation script to compare two implementations of conditional directed information:
1. The implementation in interfere/causal_information.py
2. The implementation in interfere/conditional_directed_information.py

This script uses the evaluation scenarios from interfere/causal_information.py
to generate synthetic data with known causal relationships and confounding,
then compares the results from both implementations.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from interfere.causal_information import (
    conditional_directed_information,
    _scenario_direct_only,
    _scenario_confounded,
    _scenario_mixed,
    _scenario_null
)
from interfere.conditional_directed_information import directed_information
from interfere.interventions import SignalIntervention
import random


def evaluate_implementations(n_runs=10, n_perm=1000, alpha=0.2, seed=42):
    """Compare both implementations on different scenarios."""
    print("\n" + "="*80)
    print("Starting evaluation of conditional directed information implementations")
    print("="*80)
    
    # Set parameters
    T = 800  # Time series length
    rng = np.random.default_rng(seed=seed)
    
    # Define scenarios
    scenarios = {
        "direct": lambda: _scenario_direct_only(T, a=0.9, rng=rng),
        "confounded": lambda: _scenario_confounded(T, b=0.9, c=0.9, rng=rng),
        "mixed": lambda: _scenario_mixed(T, a=0.7, b=0.5, c=0.5, rng=rng),
        "null": lambda: _scenario_null(T, rng=rng)
    }
    
    results = []
    
    for scenario_name, data_gen in scenarios.items():
        print(f"\n{'='*40}")
        print(f"Evaluating scenario: {scenario_name}")
        print(f"{'='*40}")
        
        # Generate data
        X, Y, Z = data_gen()
        print(f"Generated data with shapes: X={X.shape}, Y={Y.shape}, Z={Z.shape if Z is not None else None}")
        
        # Create intervention data
        print("\nCreating intervention data...")
        if scenario_name == "direct":
            # For direct causality, intervene on Y
            X_int = X.copy()
            Y_int = np.ones_like(Y)  # Fix Y to 1.0
            Z_int = None
        elif scenario_name == "confounded":
            # For confounded scenario, intervene on Y
            X_int = X.copy()
            Y_int = np.ones_like(Y)  # Fix Y to 1.0
            Z_int = Z.copy()
        elif scenario_name == "mixed":
            # For mixed scenario, intervene on Y
            X_int = X.copy()
            Y_int = np.ones_like(Y)  # Fix Y to 1.0
            Z_int = Z.copy()
        else:  # null
            # For null scenario, intervene on Y
            X_int = X.copy()
            Y_int = np.ones_like(Y)  # Fix Y to 1.0
            Z_int = None
        
        print("✓ Intervention data created")
        
        # Method 1: Using conditional_directed_information from causal_information.py
        print("\nApplying Method 1 (conditional_directed_information)...")
        
        # Reshape arrays to be 1-D and ensure they have the same length
        max_lag = 1  # Same as in the function
        X_1d = X[max_lag:].reshape(-1)  # Remove first max_lag points to match function's behavior
        Y_1d = Y[:-max_lag].reshape(-1)  # Remove last max_lag points to match function's behavior
        
        # Normalize the data for better discretization
        X_1d = (X_1d - X_1d.mean()) / X_1d.std()
        Y_1d = (Y_1d - Y_1d.mean()) / Y_1d.std()
        
        # For the direct and null scenarios, we need to create a dummy Z array
        if Z is None:
            # Create a dummy Z array with the same shape as X and Y
            Z_1d = np.zeros_like(X_1d)
            result1 = conditional_directed_information(
                X_1d, Y_1d, Z_1d,
                max_lag=max_lag,
                n_perm=n_perm,
                alpha=alpha,
                n_bins=32,  # Further increased number of bins for better discretization
                rng=random.Random(rng.integers(1 << 30))
            )
        else:
            Z_1d = Z[max_lag:].reshape(-1)  # Remove first max_lag points to match X_1d
            # Normalize Z as well
            Z_1d = (Z_1d - Z_1d.mean()) / Z_1d.std()
            result1 = conditional_directed_information(
                X_1d, Y_1d, Z_1d,
                max_lag=max_lag,
                n_perm=n_perm,
                alpha=alpha,
                n_bins=32,  # Further increased number of bins for better discretization
                rng=random.Random(rng.integers(1 << 30))
            )
        
        print(f"✓ Method 1 result: {result1}")
        
        # Method 2: Using directed_information from conditional_directed_information.py
        print("\nApplying Method 2 (directed_information)...")
        # Reshape arrays for Method 2
        X_2d = X.reshape(-1, 1)
        Y_2d = Y.reshape(-1, 1)
        Y_int_2d = Y_int.reshape(-1, 1)
        
        # Normalize the data for Method 2, handling constant intervention data
        X_2d = (X_2d - X_2d.mean()) / (X_2d.std() + 1e-10)
        Y_2d = (Y_2d - Y_2d.mean()) / (Y_2d.std() + 1e-10)
        Y_int_2d = np.zeros_like(Y_int_2d)  # Set intervention to zero mean
        
        result2 = directed_information(X_2d, Y_2d, Y_int_2d)
        print(f"✓ Method 2 result: {result2:.4f}")
        
        # Store results
        results.append({
            'scenario': scenario_name,
            'method1_result': result1,
            'method2_score': result2,
            'expected_result': {
                'direct': 'direct_or_unblocked',
                'confounded': 'confounded',
                'mixed': 'direct_or_unblocked',
                'null': 'no_dependence'
            }[scenario_name]
        })
        
        # Print interpretation
        print("\nInterpretation:")
        print(f"{'='*40}")
        if result1 == results[-1]['expected_result']:
            print(f"✓ Method 1 correctly identified the scenario as {result1}")
        else:
            print(f"✗ Method 1 incorrectly identified the scenario as {result1} (expected {results[-1]['expected_result']})")
        
        # For method 2, we expect higher scores for direct/mixed scenarios and lower for confounded/null
        if scenario_name in ['direct', 'mixed']:
            if result2 > 0.5:  # Threshold for normalized data
                print(f"✓ Method 2 detected significant directed information ({result2:.4f})")
            else:
                print(f"✗ Method 2 failed to detect significant directed information ({result2:.4f})")
        else:  # confounded or null
            if result2 < 0.5:  # Threshold for normalized data
                print(f"✓ Method 2 correctly detected low directed information ({result2:.4f})")
            else:
                print(f"✗ Method 2 incorrectly detected high directed information ({result2:.4f})")
    
    return results


def plot_results(results):
    """Plot comparison results."""
    print("\nGenerating comparison plots...")
    
    # Create images directory if it doesn't exist
    os.makedirs('images', exist_ok=True)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot scores for both methods
    scenarios = [r['scenario'] for r in results]
    method2_scores = [r['method2_score'] for r in results]
    
    x = np.arange(len(scenarios))
    width = 0.35
    
    ax1.bar(x, method2_scores, width, label='Method 2 (directed_information)')
    
    ax1.set_xlabel('Scenario')
    ax1.set_ylabel('Score')
    ax1.set_title('Directed Information Scores')
    ax1.set_xticks(x)
    ax1.set_xticklabels(scenarios)
    ax1.legend()
    
    # Plot accuracy of method 1
    method1_correct = [1 if r['method1_result'] == r['expected_result'] else 0 for r in results]
    
    ax2.bar(x, method1_correct, width, label='Method 1 (conditional_directed_information)')
    
    ax2.set_xlabel('Scenario')
    ax2.set_ylabel('Correct (1) / Incorrect (0)')
    ax2.set_title('Accuracy of Confounding Detection')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Incorrect', 'Correct'])
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('images/conditional_directed_information_comparison.png')
    plt.close()
    print("✓ Plots generated and saved to 'images/conditional_directed_information_comparison.png'")


def main():
    """Main function to run the evaluation."""
    print("\n" + "="*80)
    print("Starting evaluation of conditional directed information implementations")
    print("="*80)
    
    try:
        results = evaluate_implementations(n_runs=10, n_perm=1000, alpha=0.2)
        plot_results(results)
        print("\n" + "="*80)
        print("Evaluation completed successfully!")
        print("="*80)
    except Exception as e:
        print("\n" + "="*80)
        print("Error during evaluation:")
        print(f"{str(e)}")
        print("="*80)
        raise


if __name__ == "__main__":
    main() 