#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load Python/3.9.5-GCCcore-10.3.0
module load FFmpeg/4.3.2-GCCcore-10.3.0

source speechbrain_venv/bin/activate

python _TEAM2_speechbrain_gpu_setup_test.py