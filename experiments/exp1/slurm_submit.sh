#!/bin/sh
 
# Expected number of hours to run one simulation (always overestimate so that slurm doesnt kill the sim)
HOURS_PER_SIM=24
# Directory to store the output data
DATADIR="/work/users/d/j/djpassey/interfere_exp1"

# Arguments for sbatch. Sets the appropriate time limit and directory
FLAGS="--ntasks=1 --cpus-per-task=1 --time=$HOURS_PER_SIM:00:00 --chdir=$DATADIR"

# Total number of jobs
NJOBS=300
for((n=0; n<$NJOBS; n+=1)); do
    # Submit the multiple parameter job script to the clusters
    sbatch $FLAGS /nas/longleaf/home/djpassey/interfere/experiments/exp1/single_slurm_job.sh $n
done