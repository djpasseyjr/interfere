#!/bin/sh

# Expected number of hours to run one simulation (always overestimate so that slurm doesnt kill the sim)
HOURS_PER_SIM=48
GIGS=32

# Arguments for sbatch. Sets the appropriate time limit and directory
FLAGS="--ntasks=1 --mem=${GIGS}G  --cpus-per-task=1 --time=$HOURS_PER_SIM:00:00 --chdir=$DATADIR"
# Total number of jobs
NJOBS=$(python3 -c "import exp_tools as ex; ex.print_num_jobs()")
for((n=0; n<$NJOBS; n+=1)); do

    incomplete=$(python3 -c "import exp_tools as ex; ex.is_incomplete("$n")")

    if [[ "$incomplete" == 'True' ]]; then
        # Submit the multiple parameter job script to the clusters
        sbatch $FLAGS /nas/longleaf/home/djpassey/interfere/experiments/exp1/runner/single_slurm_job.sh $n
        echo "Submitted exp_output$n.ipynb"
    else
        echo "exp_output$n.ipynb is NOT SUBMITTED" 
    fi

done
