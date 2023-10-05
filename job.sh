#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --error=%j.err_
#SBATCH --output=%J.out_
#SBATCH --cpus-per-task=32     # specify the number of CPU cores
#SBATCH --mem=64G     # specify memory per CPU


source /etc/profile.d/modules.sh

module load Python

pip3 install -q kaggle
pip3 install -q pandas matplotlib seaborn scikit-learn imbalanced-learn


python3 /home/abbass12/Desktop/CICIDS/NIDS_CICIDS18.py