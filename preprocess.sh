#!/bin/bash
#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem=64gb
#SBATCH --time=02:00:00
#SBATCH --job-name=Preprocessing
#SBATCH --error=Outputs/%j.err_
#SBATCH --output=Outputs/%J.out_
#SBATCH -p short

apptainer exec --nv tensorflow_latest-gpu.sif pip3 install -q kaggle --user
apptainer exec --nv tensorflow_latest-gpu.sif pip3 install -q matplotlib seaborn scikit-learn imbalanced-learn --user

apptainer exec --nv tensorflow_latest-gpu.sif python3 ./preprocess_data.py