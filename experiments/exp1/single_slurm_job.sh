#!/bin/sh

module purge
module load python/3.9.6
pip install --upgrade pip
pip install sktime torch pysindy
pip install /nas/longleaf/home/djpassey/interfere/

python /nas/longleaf/home/djpassey/interfere/experiments/exp1/generate_data.py $1