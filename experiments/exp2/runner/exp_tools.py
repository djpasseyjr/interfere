"""Tools for saving progress and picking up where the last experiment ended."""

import os
from pathlib import Path
import pickle as pkl
from typing import Any, Dict, List

import interfere
import numpy as np

PARAM_DIR = Path(__file__).parents[1] / "parameters"
DYN_PARAM_FNAME = "dynamic_models"
METH_PARAM_FNAME = "inference_methods"
FILE_PREFIX = "exp2"
SAVE_DIR = Path("/work/users/d/j/djpassey/interfere_exp2.0")

def save_file_path(idx):
    return SAVE_DIR / Path(f"{FILE_PREFIX}_output{idx}.pkl")

def is_incomplete(idx: str):
    i = int(idx)
    dyn_ps, _ = load_parameters()
    results = load_results(i, dyn_ps[i])
    print(not results.get("complete", False))


def print_num_jobs():
    dyn_ps, _ = load_parameters()
    print(len(dyn_ps))


def load_parameters(test=False):
    """Loads parameter files for the experiment."""
    ext = ".pkl"
    if test:
        ext = "_small.pkl"

    with open(PARAM_DIR / (DYN_PARAM_FNAME + ext),  "rb") as f:
        dyn_ps = pkl.load(f)

    with open(PARAM_DIR / (METH_PARAM_FNAME + ext), "rb") as f:
        meth_ps = pkl.load(f)

    return dyn_ps, meth_ps


def save_results_dict(results_dict, outfile_idx: str,):
    # Path to results file.
    p = save_file_path(outfile_idx)
    # Load the output pickle file
    with open(p, "wb") as f:
            pkl.dump(results_dict, f)

def check_for_existing_results(outfile_idx: str):
    """Loads a pickle object or returns None if the file doesn't exists."""

    # Check for experiment output file
    p = save_file_path(outfile_idx)
    if not p.exists():
        return None
    
    # Load the output pickle file
    with open(p, "rb") as f:
        output = pkl.load(f)
    return output


def load_results(outfile_idx: str, dynamic_model_args: dict):
    """Loads existing results or makes a new template dict"""

    results = check_for_existing_results(outfile_idx)

    if results is None:
        model_name = dynamic_model_args["model_type"].__name__
        results = {
            "methods": {},
            "dynamic_sim_complete": False,
            "complete": False,
            **dynamic_model_args
        }

    return results


def store_dynamic_model_outputs(
    Xs: List[np.ndarray],
    X_dos: List[np.ndarray],
    t: np.ndarray,
    results_dict: dict
):
    """Saves output of a dynamic model simulation."""
    results_dict["Xs"] = Xs
    results_dict["X_dos"] = X_dos
    results_dict["t"] = t
    results_dict["dynamic_sim_complete"] = True


def load_dynamic_sim(results_dict):
    """Load the output of a dynamic model simulation."""
    Xs = results_dict["Xs"] 
    X_dos = results_dict["X_dos"]
    t = results_dict["t"]
    return Xs, X_dos, t

def remove_extra_simulation_args(dyn_args):
    """Removes generate_counterfactual_forecasts incompatable kwargs.
    
    The function dyn_args was redesigned to carry additional information about a
    system in it's keyword args that is then included in the results dictionary.
    This function removes those extra args so that the dict can be passed to
    generate_counterfactual_forecasts(**arg_dict).
    """
    gcf = interfere.generate_counterfactual_forecasts
    # Grab function's number of of args.
    n_args = gcf.__code__.co_argcount
    # Grab all arg names
    arg_names = gcf.__code__.co_varnames[:n_args]
    # Make a dict containing only function args
    arg_dict = {k:dyn_args[k] for k in arg_names if k in dyn_args}
    return arg_dict


def run_dynamics(dyn_args: dict, results_dict: dict, outfile_idx: str):
    """Checks for existing sim data, and if none exists, runs the simulation."""
    
    if not results_dict["dynamic_sim_complete"]:
        Xs, X_dos, t = interfere.generate_counterfactual_forecasts(
            **remove_extra_simulation_args(dyn_args))
        store_dynamic_model_outputs(Xs, X_dos, t, results_dict)
        save_results_dict(results_dict, outfile_idx)
    else:
        Xs, X_dos, t = load_dynamic_sim(results_dict)

    # Initialize intervention (to be used by forecast methods).
    intervention = dyn_args["intervention_type"](
        **dyn_args["intervention_params"])

    return Xs, X_dos, t, intervention


def run_forecasts(
    Xs: List[np.ndarray],
    X_dos: List[np.ndarray],
    t: np.ndarray, 
    intervention: interfere.interventions.ExogIntervention,
    method_args: dict,
    results_dict: dict,
    outfile_idx: str,
    opt_all: bool = True
):
    """Accepts the output of `run_dynamics` along with `method_args` a
    dictionary of inference method arguments, the results_dict, and outfile_idx.

    The `opt_all` argument controls whether hyperparameter optimization
    happens once, for the first simulation, and those parameters are used for
    every other prediction, or if it is performed for every realization. Setting
    `opt_all=True` is much slower but better mimics real world scenarios.
    """
    # Load current progress on the method
    name = method_args["method_type"].__name__
    method_progress = load_method_progress(name, results_dict)

    # Exit if already complete
    if method_progress["complete"]:
        return None
    
    # Loads past predictions.
    method_X_do_preds = method_progress["X_do_preds"]
    method_X_preds = method_progress["X_preds"]

    # Initialize identity intervention.
    no_intervention = interfere.interventions.IdentityIntervention()

    start = len(method_X_do_preds)
    end = len(Xs)

    best_params = None
    if (start > 0) and (not opt_all):
        # Use existing parameters when not optimizing each iter.
        best_params = method_progress["best_params"]

    for i in range(start, end):
        # Compute forecast and store.
        X_preds, best_params = interfere.benchmarking.forecast_intervention(
            Xs[i], X_dos[i], t, no_intervention, **method_args,
            best_params=best_params
        )
        method_X_preds.append(X_preds)

        # Forecast the intervention response and store.
        X_do_preds, _ = interfere.benchmarking.forecast_intervention(
            Xs[i], X_dos[i], t, intervention, **method_args, best_params=best_params)
        method_X_do_preds.append(X_do_preds)

        # Save optimized hyper parameters.
        if opt_all:
            # This resets best_params to none so hyper parameter tuning is done
            # every iteration within forecast_intervention(). When opt_all is 
            # false, optimization is only done for the first iteration.
            method_progress["best_params_list"].append(best_params)
            best_params = None

        # Store progress and write to file every five time series.
        if i % 5 == 0:
            method_progress["best_params"] = best_params
            save_results_dict(results_dict, outfile_idx)

    # Mark complete and save
    method_progress["complete"] = True
    save_results_dict(results_dict, outfile_idx)
    
    
def load_method_progress(method_name: str, results_dict: dict):
    """Loads historic method progress or makes empty progress tracker if
    none."""

    # Check that the results dict has a methods key and add one if not.
    methods = results_dict.get("methods", {})
    if methods == {}:
        results_dict["methods"] = methods

    meth_progress = methods.get(method_name, None)

    # If no progress has been recorded start a new method progress and store.
    if meth_progress is None:
        meth_progress = method_progress_templ(method_name)
        results_dict["methods"][method_name] = meth_progress
    return meth_progress


def method_progress_templ(method_name):
    """Template method progress dict"""
    return {
        "complete": False,
        "best_params": None,
        "X_do_preds": [],
        "X_preds": [],
        "best_params_list": []
    }


def check_consistency(dyn_args: dict, exp_idx: int, opt_all: bool):
    """Checks outfile to make sure everything is correct."""
    results = load_results(exp_idx, dyn_args)
    dyn_params_match(results, dyn_args, exp_idx)
    
    # Get stored methods
    methods = results.get("methods", {})

    # If output contains no dynamics, ensure that there are no methods either.
    if ("Xs" not in results) and (len(methods) > 0):
        raise ValueError(
            "Output file contains method predictions but no dynamics."
            f" Command line arg: {exp_idx}")
    
    # Check that experiment matches previous opt_all settings.
    if opt_all and (len(methods) > 0):
        for k in methods.keys():
            if methods[k]["best_params"] is not None:
                raise ValueError("Single hyperparameter set found when"
                                 " OPT_ALL is set to true. Should only be one."
                                 f" Command line arg: {exp_idx}")
            
    if not opt_all and (len(methods) > 0):
        for k in methods.keys():
            if len(methods[k]["best_params_list"]) > 0:
                raise ValueError(
                    "Many hyperparameter sets found when OPT_ALL is "
                    "set to false. Should be only one set."
                    f" Command line arg: {exp_idx}")
            

def args_are_equal(arg1, arg2):
    
    if isinstance(arg1, (float, str, int)):
        return arg1 == arg2
    
    if isinstance(arg1, (list, tuple)):
        if len(arg1) != len(arg2):
            return False
        elements_are_equal = [
            args_are_equal(a1, a2) for a1, a2 in zip(arg1, arg2)
        ]
        return all(elements_are_equal)
     
    if isinstance(arg1, np.ndarray):
        return np.all(arg1 == arg2)
    
    if arg1 is None:
        return arg2 is None
    
    if callable(arg1):
        return callable(arg2)
    
    elif getattr(arg1, "__dict__", False):
        attribs_are_equal = [
            args_are_equal(arg1.__dict__[k], arg2.__dict__[k])
            for k in arg1.__dict__.keys()
        ]
        return all(attribs_are_equal)
    
    return False


def dyn_params_match(result_dict, dyn_args, exp_idx):

    # Check that the parameters of the dynamic model match the save file.
    for k in dyn_args["model_params"].keys():
        if not args_are_equal(
            dyn_args["model_params"][k],
            result_dict["model_params"][k]
        ):
            raise ValueError(
                "Stored dynamic model does not match loaded dynamic model. "
                f"Command line index arg: {exp_idx}"
            )


def finish(results, outfile_idx):
    """Wrap up."""
    results["complete"] = True
    save_results_dict(results, outfile_idx)
