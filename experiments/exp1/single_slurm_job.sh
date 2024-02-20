#!/bin/sh

module purge
module load python/3.9.6
pip install virtualenv

virtualenv exp1_venv
pip install --upgrade pip
pip install /nas/longleaf/home/djpassey/interfere/

python /nas/longleaf/home/djpassey/interfere/experiments/exp1/generate_data.py $1
