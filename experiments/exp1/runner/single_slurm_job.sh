#!/bin/sh

module purge
module load anaconda
conda activate interfere

python /nas/longleaf/home/djpassey/interfere/experiments/exp1/runner/run.py $1
