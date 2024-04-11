import os
from pathlib import Path
import pickle as pkl

import interfere
import numpy as np

# Import local file exp_tools.py
import exp_tools

EXP_IDX = "0"
OUTFILE = exp_tools.save_file_path(EXP_IDX)

DIM = 3
RNG = np.random.default_rng(11)
DYN_ARGS = dict(
    model_type=interfere.dynamics.coupled_map_1dlattice_chaotic_brownian,
    model_params={
        "dim": DIM,
        "sigma": 0.0,
        "measurement_noise_std": 0.05 * np.ones(DIM)
    },
    intervention_type=interfere.PerfectIntervention,
    intervention_params={"intervened_idxs": 0, "constants": -0.5},
    initial_conds=[RNG.random(DIM), RNG.random(DIM)],
    start_time=0, end_time=100, dt=1,
    rng = RNG
)

def cleanup():
    if OUTFILE.exists():
        os.remove(OUTFILE)



def test_result_load_and_save():

    cleanup()

    # Not outfile should exist
    assert not OUTFILE.exists()

    results = exp_tools.load_results(EXP_IDX, DYN_ARGS)

    # Check that the dict has the correct format
    for key in ["dynamic_sim_complete", "methods", "model_type"]:
        assert key in results

    # Check that progress was initialized correctly
    assert results["dynamic_sim_complete"] == False
    assert results["methods"] == {}

    results["dynamic_sim_complete"] = True
    results["complete"] = True

    exp_tools.save_results_dict(results, EXP_IDX)

    # Check that the file was written.
    assert OUTFILE.exists()
    
    results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    assert results["complete"]

    cleanup()


def test_load_params():
    d, m = exp_tools.load_parameters()
    d_small, m_small = exp_tools.load_parameters(test=True)
    assert d != d_small
    assert m != m_small
    assert "model_type" in d[0]
    assert "method_type" in m[0]
    assert "model_type" in d_small[0]
    assert "method_type" in m_small[0]


def test_run_dynamics():
    cleanup()

    dyn_args_list, _ = exp_tools.load_parameters(test=True)
    dyn_args = dyn_args_list[0]

    results = exp_tools.load_results(EXP_IDX, dyn_args)
    assert not results["dynamic_sim_complete"]

    Xs, X_dos, t, g = exp_tools.run_dynamics(dyn_args, results, EXP_IDX)

    # Check that sim was marked complete.
    assert results["dynamic_sim_complete"]
    for key in ["Xs", "X_dos", "t"]:
        assert key in results

    # Check that the results were saved
    reload_results = exp_tools.load_results(EXP_IDX, dyn_args)
    assert reload_results["dynamic_sim_complete"]
    for key in ["Xs", "X_dos", "t"]:
        assert key in reload_results

    _Xs, _, _, _ = exp_tools.run_dynamics(dyn_args, results, EXP_IDX)
    assert np.all(_Xs == Xs)

    cleanup()

    

def test_load_method_progress():
    cleanup()

    dyn_args_list, method_arg_list = exp_tools.load_parameters(test=True)
    dyn_args = dyn_args_list[0]
    results = exp_tools.load_results(EXP_IDX, dyn_args)
    Xs, X_dos, t, g = exp_tools.run_dynamics(dyn_args, results, EXP_IDX)

    method_name = method_arg_list[0]["method_type"].__name__
    assert method_name not in results["methods"]

    progress = exp_tools.load_method_progress(method_name, results)
    assert method_name in results["methods"]

    for key in ["complete", "best_params", "X_do_preds", "best_params_list"]:
        assert key in progress
    assert not progress["complete"]
    assert progress["best_params"] is None

    # Alter the progress dict only (not main result dict) and save.
    progress["complete"] = True
    exp_tools.save_results_dict(results, EXP_IDX)

    reload_progress = exp_tools.load_method_progress(method_name, results)
    assert reload_progress["complete"]

    cleanup()


def test_forecast():

    cleanup()

    _, method_arg_list = exp_tools.load_parameters(test=True)
    results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    dyn_sim_output = exp_tools.run_dynamics(DYN_ARGS, results, EXP_IDX)

    # Loop over each infernce method.
    for margs in method_arg_list:
        # Tune hyper parameters, run forecasts and store results.
        exp_tools.run_forecasts(
            *dyn_sim_output, margs, results, EXP_IDX, opt_all=False)
        
    # Make sure experiment was saved
    assert len(results["methods"]) == len(method_arg_list)
    _results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    assert len(_results["methods"]) == len(method_arg_list)

    # Make sure the correct number of exeriments ran
    Xs = dyn_sim_output[0]
    for margs in method_arg_list:
        name = margs["method_type"].__name__
        progress = exp_tools.load_method_progress(name, results)
        reload_progress = exp_tools.load_method_progress(name, _results)
        for p in [progress, reload_progress]:

            assert p["complete"] == True
            assert len(Xs) == len(p["X_do_preds"])
            assert len(p["best_params_list"]) == 0
            assert p["best_params"] is not None

    cleanup()


def test_forecast_opt_all():

    cleanup()

    _, method_arg_list = exp_tools.load_parameters(test=True)
    results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    dyn_sim_output = exp_tools.run_dynamics(DYN_ARGS, results, EXP_IDX)

    # Loop over each infernce method.
    for margs in method_arg_list:
        # Tune hyper parameters, run forecasts and store results.
        exp_tools.run_forecasts(
            *dyn_sim_output, margs, results, EXP_IDX, opt_all=True)
        
    # Make sure experiment was saved
    assert len(results["methods"]) == len(method_arg_list)
    _results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    assert len(_results["methods"]) == len(method_arg_list)

    # Make sure the correct number of exeriments ran
    Xs = dyn_sim_output[0]
    for margs in method_arg_list:
        name = margs["method_type"].__name__
        progress = exp_tools.load_method_progress(name, results)
        reload_progress = exp_tools.load_method_progress(name, _results)
        for p in [progress, reload_progress]:

            assert p["complete"] == True
            assert len(Xs) == len(p["X_do_preds"])
            assert len(Xs) == len(p["best_params_list"])
            assert p["best_params"] is None


    cleanup()


def test_partial_forecast():

    cleanup()

    _, method_arg_list = exp_tools.load_parameters(test=True)
    results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    dyn_sim_output = exp_tools.run_dynamics(DYN_ARGS, results, EXP_IDX)

    # Loop over first infernce methods.
    for margs in method_arg_list[:2]:
        # Tune hyper parameters, run forecasts and store results.
        exp_tools.run_forecasts(
            *dyn_sim_output, margs, results, EXP_IDX, opt_all=False)
        
    # Make sure that one of the methods was not run.
    not_run_method = method_arg_list[-1]["method_type"].__name__
    assert not_run_method not in results["methods"]
        
    # Remove all but one prediction for the last method (VAR).
    first_method = "VAR"
    X_do_preds = results["methods"][first_method]["X_do_preds"]
    X_preds = results["methods"][first_method]["X_preds"]

    results["methods"][first_method]["X_do_preds"] = X_do_preds[:1]
    results["methods"][first_method]["X_preds"] = X_preds[:1]

    # Change method to unfinished.
    results["methods"][first_method]["complete"] = False
    orig_best_params = results["methods"][first_method]["best_params"]
    assert orig_best_params is not None

    exp_tools.save_results_dict(results, EXP_IDX)


    # Rerun experiment.
    _, method_arg_list = exp_tools.load_parameters(test=True)
    results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    dyn_sim_output = exp_tools.run_dynamics(DYN_ARGS, results, EXP_IDX)


    # Only one forecast existed for `first_method`
    assert len(results["methods"][first_method]["X_do_preds"]) == 1
    assert len(results["methods"][first_method]["X_preds"]) == 1

    # Loop over ALL infernce methods.
    for margs in method_arg_list:
        # Tune hyper parameters, run forecasts and store results.
        exp_tools.run_forecasts(
            *dyn_sim_output, margs, results, EXP_IDX, opt_all=False)
        
    # After running the forecasts, more forecasts exists for `first_method`.
    assert len(results["methods"][first_method]["X_do_preds"]) > 1
    assert len(results["methods"][first_method]["X_preds"]) > 1
        
    assert orig_best_params == results["methods"][first_method]["best_params"]

    # Check that half finished method data was preserved.
    X_do_pred_old = X_do_preds[0][0]
    X_pred_old = X_preds[0][0]
    assert np.all(
        results["methods"][first_method]["X_do_preds"][0][0] == X_do_pred_old)
    assert np.all(
        results["methods"][first_method]["X_preds"][0][0] == X_pred_old)
    

    # Make sure the correct number of exeriments ran
    Xs = dyn_sim_output[0]
    for margs in method_arg_list:
        name = margs["method_type"].__name__
        progress = exp_tools.load_method_progress(name, results)
        assert progress["complete"] == True
        assert len(Xs) == len(progress["X_do_preds"])
        assert len(Xs) == len(progress["X_preds"])
        assert len(progress["best_params_list"]) == 0
        assert progress["best_params"] is not None

    cleanup()


def test_check_consistency():

    cleanup()
    exp_tools.check_consistency(DYN_ARGS, EXP_IDX, opt_all=False)    
    cleanup()

    _, method_arg_list = exp_tools.load_parameters(test=True)
    results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    
    exp_tools.check_consistency(DYN_ARGS, EXP_IDX, opt_all=False)    

    dyn_sim_output = exp_tools.run_dynamics(DYN_ARGS, results, EXP_IDX)

    exp_tools.check_consistency(DYN_ARGS, EXP_IDX, opt_all=False)    

    # Loop over 2/3 infernce methods.
    for margs in method_arg_list[:2]:
        # Tune hyper parameters, run forecasts and store results.
        exp_tools.run_forecasts(
            *dyn_sim_output, margs, results, EXP_IDX, opt_all=False)

    exp_tools.check_consistency(DYN_ARGS, EXP_IDX, opt_all=False)    
        
    # Remove all but one prediction for the last method (VAR).
    first_method = "VAR"
    X_do_preds = results["methods"][first_method]["X_do_preds"]
    results["methods"][first_method]["X_do_preds"] = X_do_preds[:1]

    # Change method to unfinished.
    results["methods"][first_method]["complete"] = False

    exp_tools.save_results_dict(results, EXP_IDX)


    # Rerun experiment.
    _, method_arg_list = exp_tools.load_parameters(test=True)
    results = exp_tools.load_results(EXP_IDX, DYN_ARGS)
    exp_tools.check_consistency(DYN_ARGS, EXP_IDX, opt_all=False)    

    dyn_sim_output = exp_tools.run_dynamics(DYN_ARGS, results, EXP_IDX)
    exp_tools.check_consistency(DYN_ARGS, EXP_IDX, opt_all=False)    


    # Loop over ALL infernce methods.
    for margs in method_arg_list:
        # Tune hyper parameters, run forecasts and store results.
        exp_tools.run_forecasts(
            *dyn_sim_output, margs, results, EXP_IDX, opt_all=False)
        
    exp_tools.check_consistency(DYN_ARGS, EXP_IDX, opt_all=False)    
    cleanup()


def test_args_are_equal():

    # Numbers
    assert exp_tools.args_are_equal(1, 1.0)
    assert not exp_tools.args_are_equal(1, 2)

    # Strings
    assert exp_tools.args_are_equal("words", "words")
    assert not exp_tools.args_are_equal("words", "word")

    # Arrays
    assert exp_tools.args_are_equal(np.ones((2, 2)), np.ones((2, 2)))
    assert not exp_tools.args_are_equal(np.ones((2, 2)), np.zeros((2, 2)))

    # Classes
    m0 = interfere.dynamics.coupled_map_1dlattice_chaotic_brownian()
    m1 = interfere.dynamics.coupled_map_1dlattice_chaotic_brownian(dim=10)
    assert exp_tools.args_are_equal(m0, m1)
    
    m2 = interfere.dynamics.coupled_map_1dlattice_chaotic_traveling_wave()
    assert not exp_tools.args_are_equal(m0, m2)

    # None
    assert exp_tools.args_are_equal(None, None)
    assert not exp_tools.args_are_equal(None, "xxx")
    
