"""
Tests for the conditional directed information implementation.
"""

import numpy as np
import pytest
from interfere.causal_information import DirectedInformation, ConditionalDirectedInformationMetric
from interfere.interventions import SignalIntervention


def test_directed_information_basic():
    """Test basic directed information calculation."""
    # Create a simple time series with a causal relationship
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    
    # X causes Y with some noise
    X = np.sin(t).reshape(-1, 1)
    Y = 2 * X + 0.1 * np.random.randn(n_samples, 1)
    
    # Calculate directed information
    di = DirectedInformation()
    di_value = di.compute_directed_information(X, Y)
    
    # Should be positive (X causes Y)
    assert di_value > 0
    
    # Reverse direction should be smaller
    di_value_reverse = di.compute_directed_information(Y, X)
    assert di_value > di_value_reverse


def test_conditional_directed_information():
    """Test conditional directed information calculation."""
    # Create a time series with a confounder
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    
    # Z is a confounder that affects both X and Y
    Z = np.sin(2 * t).reshape(-1, 1)
    X = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    Y = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    
    # Create interventional data by fixing X
    X_int = np.ones((n_samples, 1))  # Fixed value
    Y_int = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    
    # Calculate conditional directed information
    di = DirectedInformation()
    di_value = di.compute_conditional_directed_information(X, Y, X_int, Y_int, Z)
    
    # Should be positive (confounding is present)
    assert di_value > 0


def test_confounding_detection():
    """Test that confounding can be detected using the metric."""
    # Create a time series with a confounder
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    
    # Z is a confounder that affects both X and Y
    Z = np.sin(2 * t).reshape(-1, 1)
    X = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    Y = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    
    # Create interventional data by fixing X
    X_int = np.ones((n_samples, 1))  # Fixed value
    Y_int = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    
    # Use the metric to detect confounding
    metric = ConditionalDirectedInformationMetric(threshold=0.05)
    
    # Combine data for the metric
    X_combined = np.hstack([X, Y, Z])
    X_int_combined = np.hstack([X_int, Y_int, Z])
    
    # Create a dummy prediction (not used in confounding detection)
    X_do_pred = X_int_combined.copy()
    
    # Specify intervention indices (X is intervened)
    intervention_idxs = [0]
    
    # Calculate confounding score
    result = metric(X_combined, X_int_combined, X_do_pred, intervention_idxs)
    
    # Check that confounding was detected
    assert len(result["confounding"]) > 0
    
    # Check that the confounding involves the confounder (Z)
    confounder_idx = 2  # Z is the third variable
    has_confounder = any((i == confounder_idx or j == confounder_idx) 
                         for i, j in result["confounding"].keys())
    assert has_confounder


def test_no_confounding_detection():
    """Test that no confounding is detected when there is none."""
    # Create a time series with a direct causal relationship (no confounder)
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    
    # X directly causes Y
    X = np.sin(t).reshape(-1, 1)
    Y = 2 * X + 0.1 * np.random.randn(n_samples, 1)
    
    # Create interventional data by fixing X
    X_int = np.ones((n_samples, 1))  # Fixed value
    Y_int = 2 * X_int + 0.1 * np.random.randn(n_samples, 1)
    
    # Use the metric to detect confounding
    metric = ConditionalDirectedInformationMetric(threshold=0.05)
    
    # Combine data for the metric
    X_combined = np.hstack([X, Y])
    X_int_combined = np.hstack([X_int, Y_int])
    
    # Create a dummy prediction (not used in confounding detection)
    X_do_pred = X_int_combined.copy()
    
    # Specify intervention indices (X is intervened)
    intervention_idxs = [0]
    
    # Calculate confounding score
    result = metric(X_combined, X_int_combined, X_do_pred, intervention_idxs)
    
    # Check that no confounding was detected
    assert len(result["confounding"]) == 0


def test_with_interfere_intervention():
    """Test confounding detection with an actual Interfere intervention."""
    # Create a dynamic system with a confounder
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
    
    # Use the metric to detect confounding
    metric = ConditionalDirectedInformationMetric(threshold=0.05)
    
    # Combine data for the metric
    X_combined = np.hstack([X, Y, Z])
    X_int_combined = np.hstack([X_int, Y_int, Z])
    
    # Create a dummy prediction (not used in confounding detection)
    X_do_pred = X_int_combined.copy()
    
    # Calculate confounding score
    result = metric(X_combined, X_int_combined, X_do_pred, intervention)
    
    # Check that confounding was detected
    assert len(result["confounding"]) > 0
    
    # Check that the confounding involves the confounder (Z)
    confounder_idx = 2  # Z is the third variable
    has_confounder = any((i == confounder_idx or j == confounder_idx) 
                         for i, j in result["confounding"].keys())
    assert has_confounder


def test_compute_confounding_score():
    """Test the compute_confounding_score method."""
    # Create a time series with a confounder
    n_samples = 1000
    t = np.linspace(0, 10, n_samples)
    
    # Z is a confounder that affects both X and Y
    Z = np.sin(2 * t).reshape(-1, 1)
    X = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    Y = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    
    # Create interventional data by fixing X
    X_int = np.ones((n_samples, 1))  # Fixed value
    Y_int = 0.5 * Z + 0.1 * np.random.randn(n_samples, 1)
    
    # Use the metric to compute confounding score
    metric = ConditionalDirectedInformationMetric()
    
    # Combine data for the metric
    X_combined = np.hstack([X, Y, Z])
    X_int_combined = np.hstack([X_int, Y_int, Z])
    
    # Specify intervention indices (X is intervened)
    intervention_idxs = [0]
    
    # Calculate confounding score
    score = metric.compute_confounding_score(X_combined, X_int_combined, intervention_idxs)
    
    # Check that the score is positive (confounding is present)
    assert score > 0 