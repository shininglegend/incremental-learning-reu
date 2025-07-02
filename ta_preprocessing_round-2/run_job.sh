#!/bin/bash
source /opt/conda/bin/activate
conda remove -n ta-env --all -y
conda env create -f ../linux_env.yml -y
conda activate ta-env
python main.py
