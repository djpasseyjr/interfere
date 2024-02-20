#!/bin/sh

module purge
module load python/3.9.6
pip install virtualenv

virtualenv exp1_venv
source exp1_venv/bin/activate

pip install --upgrade pip
pip install -r /nas/longleaf/home/djpassey/interfere/experiments/exp1/requirements.txt
pip install /nas/longleaf/home/djpassey/interfere/

python /nas/longleaf/home/djpassey/interfere/experiments/exp1/generate_data.py $1
