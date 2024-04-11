#!/bin/sh

module purge
module load anaconda
conda activate interfere_exp2

python /nas/longleaf/home/djpassey/interfere/experiments/exp2/runner/run.py $1
