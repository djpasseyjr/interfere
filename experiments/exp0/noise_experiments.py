import os 
import sys

import interfere
import numpy as np
import pickle as pkl
import pysindy as ps


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FILE_PREFIX = "noise_exp"

# Parse command line argument that designates the index of the hyper parameters
exp_idx = int(sys.argv[1])

# Read in the hyper parameter array and access array at the command line arg
with open(f"{DIR_PATH}/{FILE_PREFIX}.pkl", "rb") as f:
    args = pkl.load(f)[exp_idx]

# Benchmark the system at this set of hyper parameters
scores, best_ps, X_do_preds = interfere.benchmarking.benchmark(*args)

# Store the hyper parameters along with the scores
args[1]["scores"] = scores
args[1]["best_ps"] = best_ps
args[1]["X_do_preds"] = X_do_preds

# Save this set of scores and parameters to a file whose name contains
# the index passed in the command line.
with open(f"{FILE_PREFIX}_{exp_idx}.pkl", "wb") as f:
    pkl.dump(args, f)
