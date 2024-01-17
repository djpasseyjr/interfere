import numpy as np
# TODO might be possible to drop this dep. Used to 
# initialize SindyDerivative:
import pysindy as ps  
from statsmodels.tsa.vector_ar.util import varsim

import interfere
from interfere.methods import simulate_perfect_intervention_var



def test_simulate_perfect_intervention_var():

    seed = 1
    rs = np.random.RandomState(seed)

    # Initialize a random VAR model
    A1 = rs.rand(3, 3) - 0.5
    A2 = rs.rand(3, 3) - 0.5
    coefs = np.stack([A1, A2])
    mu = rs.rand(3)
    Z = rs.rand(3, 3)
    Sigma = Z * Z.T
    steps = 101
    initial_vals = np.ones((2, 3))
    nsims = 10000

    # Simulate it
    true_var_sim = varsim(
        coefs,
        mu,
        Sigma,
        steps=steps,
        initial_values=initial_vals,
        seed=seed,
        nsimulations=nsims,
    )

    # Copy the VAR but add another dimension where the intervention will be
    # applied. 
    intervention_idx = 3
    intervention_value = 10
    intervention_coefs = 0.5 * np.stack([np.eye(4), np.eye(4)])
    intervention_coefs[:, :3, :3] = coefs
    intervention_mu = rs.rand(4)
    intervention_mu[:3] = mu
    intervention_sigma = np.eye(4)
    intervention_sigma[:3, :3] = Sigma
    intervention_initial_vals = np.ones((2, 4))


    perf_inter_sim = simulate_perfect_intervention_var(
        intervention_idx,
        intervention_value,
        intervention_coefs,
        intervention_mu,
        intervention_sigma,
        steps=steps,
        initial_values=intervention_initial_vals,
        seed=seed,
        nsimulations=nsims,
    )

    # Check that the shape is correct
    assert true_var_sim.shape == perf_inter_sim[:, :, :3].shape

    # Check that the intervention was applied correctly
    assert np.all(perf_inter_sim[:, :, intervention_idx] == intervention_value)

    # Average over the 10000 simulations to compute the expected trajectory.
    # Make sure it is equal for both models.
    assert np.all(
        np.mean(true_var_sim - perf_inter_sim[:, :, :3], axis=0) < 0.1
    )

    # Do a third simulation to  double check that the above average doesn't hold
    # in general
    A1 = rs.rand(3, 3) - 0.5
    A2 = rs.rand(3, 3) - 0.5
    coefs = np.stack([A1, A2])
    mu = rs.rand(3)
    Z = rs.rand(3, 3)
    Sigma = Z * Z.T
    steps = 101
    initial_vals = np.ones((2, 3))
    nsims = 10000

    true_var_sim2 = varsim(
        coefs,
        mu,
        Sigma,
        steps=steps,
        initial_values=initial_vals,
        seed=seed,
        nsimulations=nsims,
    )

    assert not np.all(
        np.mean(true_var_sim - true_var_sim2, axis=0) < 0.1
    )

def test_benchmark_methods():

    # Define parameters
    gen_ctrf_kwargs = dict(
        model_type=interfere.dynamics.Liping3DQuadFinance,
        model_params={"sigma": 0.5, "measurement_noise_std": np.ones(3)},
        intervention_type=interfere.PerfectIntervention,
        intervention_params={"intervened_idxs": 0, "constants": 0.1},
        initial_condition_iter=[0.01 * np.ones(3)],
        time_points_iter=[np.linspace(0, 20, 2000)],
    )
    
    # Simulate dynamic model
    X, X_do = interfere.generate_counterfactual_dynamics(**gen_ctrf_kwargs)
    # Initialize intervention
    intervention=gen_ctrf_kwargs["intervention_type"](
            **gen_ctrf_kwargs["intervention_params"])

    score_methods_kwargs = dict(
        X=X[0],
        X_do=X_do[0],
        time_points=gen_ctrf_kwargs["time_points_iter"][0],
        intervention=intervention,
        method_type=interfere.methods.SINDY,
        method_params={
            "differentiation_method": ps.SINDyDerivative(kind='spline', s=2.0)
        },
        method_param_grid={
            "optimizer__threshold": [1e-3, 1e-4],
            "differentiation_method__kwargs": [
                {'kind': 'finite_difference', 'k': 1},
                {'kind': 'spline', 's': 1.0},
                {'kind': 'spline', 's': 0.1},
            ]
        },
        num_intervention_sims=2,
        score_function=interfere.benchmarking.directional_accuracy,
        score_function_args={"intervention_idx": intervention.intervened_idxs}
    )
    
    # Score method
    score, best_params, X_do_pred = interfere.benchmarking.score_extrapolation_method(
        **score_methods_kwargs
    )


    # Change args to the VAR model
    score_methods_kwargs["method_type"] = interfere.methods.VAR
    score_methods_kwargs["method_params"] = {}
    score_methods_kwargs["method_param_grid"] = {
        "maxlags": [1, 2,],
        "trend" : ["c", "ct", "n"]
    }
    
    # Score method
    score, best_params, X_do_pred = interfere.benchmarking.score_extrapolation_method(
        **score_methods_kwargs
    )

    # Change args to the LTSFLinearForecaster model
    score_methods_kwargs["method_type"] = interfere.methods.LTSFLinearForecaster
    score_methods_kwargs["method_params"] = {"seq_len": 1, "pred_len": 1}
    score_methods_kwargs["method_param_grid"] = {
        "seq_len": [1, 2]
    }
    # Score method
    score, best_params, X_do_pred = interfere.benchmarking.score_extrapolation_method(
        **score_methods_kwargs
    )