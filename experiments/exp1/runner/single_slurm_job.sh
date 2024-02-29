#!/bin/sh

module purge
module load python/3.9.6
pip install virtualenv

virtualenv venv_exp1
source venv_exp1/bin/activate

pip install --upgrade pip
pip install -r /nas/longleaf/home/djpassey/interfere/experiments/exp1/requirements.txt
pip install git+https://www.github.com/djpasseyjr/interfere.git

python /nas/longleaf/home/djpassey/interfere/experiments/exp1/runner/run.py $1
