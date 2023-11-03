#!/bin/bash
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=60gb
#SBATCH --time=12:00:00
#SBATCH --job-name=Classification
#SBATCH --error=jupyter_err.log
#SBATCH --output=jupyter.log
#SBATCH -p tesla
#SBATCH -q tesla
#SBATCH -A kdm-grant594
#SBATCH --gres=gpu:tesla:1

apptainer exec --nv pytorch_latest.sif pip3 install -q kaggle --user
apptainer exec --nv pytorch_latest.sif pip3 install -q matplotlib seaborn scikit-learn imbalanced-learn --user
apptainer exec --nv pytorch_latest.sif pip3 install -q jupyterlab notebook
apptainer exec --nv pytorch_latest.sif pip3 install -q jupyterlab notebook --user
apptainer exec --nv pytorch_latest.sif export PATH="$PATH:/home/abbass12/.local/bin"
apptainer exec --nv pytorch_latest.sif export PATH="$PATH:/root/.local/bin"


apptainer exec --nv pytorch_latest.sif jupyter lab --ip=0.0.0.0 --port=62094 --no-browser
