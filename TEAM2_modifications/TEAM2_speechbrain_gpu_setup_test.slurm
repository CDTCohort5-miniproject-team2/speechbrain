#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2

module load Anaconda3/2022.10
module load FFmpeg/4.3.2-GCCcore-10.3.0

source activate speechbrain_venv

python _TEAM2_speechbrain_gpu_setup_test.py