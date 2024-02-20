import pickle as pkl
import os
import sys

import interfere

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FILE_PREFIX = "exp1"

# Parse command line argument that designates the index of the hyper parameters
PARAM_IDX = int(sys.argv[1])
SAVE_FILE = f"{FILE_PREFIX}_output{PARAM_IDX}.pkl"

# Read the dynamic models param array and access at the command line arg index
with open(f"{DIR_PATH}/dynamic_models_small.pkl", "rb") as f:
    dyn_args = pkl.load(f)[PARAM_IDX]

# Read in the list of inference methods
with open(f"{DIR_PATH}/inference_methods_small.pkl", "rb") as f:
    methods_args = pkl.load(f)


# Run the dynamic simulations.
Xs, X_dos, t = interfere.generate_counterfactual_forecasts(**dyn_args)

# Save dynamic model simulation data
results = {
    **dyn_args,
    "Xs": Xs,
    "X_dos": X_dos
}

with open(SAVE_FILE, "wb") as f:
    pkl.dump(results, f)


# Run forecasting methods
intervention = dyn_args["intervention_type"](**dyn_args["intervention_params"])

method_predictions = []
for ma in methods_args:
    method_preds = []
    method_best_ps = []
    for i in range(len(Xs)):
        X_do_pred, best_params = interfere.benchmarking.forecast_intervention(
            Xs[i], X_dos[i], t, intervention, **ma)
        method_preds.append(X_do_pred)
        method_best_ps.append(best_params)
    
    method_predictions.append({
        "X_do_pred": X_do_pred,
        "best_params": method_best_ps,
        **ma
    })

results["method_results"] = method_predictions
with open(SAVE_FILE, "wb") as f:
    pkl.dump(results, f)





