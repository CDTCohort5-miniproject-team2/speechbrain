#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=80G

module load Anaconda3/2022.10
module list
source activate speechbrain_venv

python ../_TEAM2_speechbrain_gpu_setup_test.py

pytest ../../tests --device='cuda'