import os 
import sys

import interfere
import numpy as np
import pickle as pkl
import pysindy as ps


DIR_PATH = os.path.dirname(os.path.realpath(__file__))
FILE_PREFIX = "noise_exp"

exp_idx = int(sys.argv[1])
with open(f"{DIR_PATH}/{FILE_PREFIX}_small.pkl", "rb") as f:
    args = pkl.load(f)[exp_idx]


scores, best_ps, X_do_preds = interfere.benchmarking.benchmark(*args)
args[1]["scores"] = scores
args[1]["best_ps"] = best_ps
args[1]["X_do_preds"] = X_do_preds


with open(f"{FILE_PREFIX}_small_{exp_idx}.pkl", "wb") as f:
    pkl.dump(args, f)
