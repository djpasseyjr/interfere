import pickle as pkl
import os
import sys

import interfere

# Import exp_tools.py from the local directory:
import exp_tools

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FILE_PREFIX = "exp1"

# Parse command line argument that designates the index of the hyper parameters.
PARAM_IDX = int(sys.argv[1])

# Save file name.
SAVE_FILE = f"{FILE_PREFIX}_output{PARAM_IDX}.pkl"

# Toggles the amount of hyper parameter optimization. See exp_tools.run_forecast
OPT_ALL = False

# Toggles a test run vs. a full run.
IS_TEST_RUN = False

# Read the dynamic models param array and access at the command line arg index.
# Read in the list of inference methods.
dyn_args_list, method_arg_list = exp_tools.load_parameters(IS_TEST_RUN) 
dyn_args = dyn_args_list[PARAM_IDX]

# Check that the previous save file (if any) is consistent with experiment.
exp_tools.check_consistency(SAVE_FILE, dyn_args, PARAM_IDX, OPT_ALL)

# Load result file from previous runs, or make an empty one.
results = exp_tools.load_results(SAVE_FILE, dyn_args)

# Run the dyanamic simulations.
dyn_sim_output = exp_tools.run_dynamics(dyn_args, results, SAVE_FILE)

# Loop over each infernce method.
for margs in method_arg_list:
    # Tune hyper parameters, run forecasts and store results.
    exp_tools.run_forecasts(
        *dyn_sim_output, margs, results, SAVE_FILE, opt_all=OPT_ALL)

# Final save.
exp_tools.finish(results, SAVE_FILE)