#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1

module load Anaconda3/2022.10
module load pytest-xdist/2.5.0-GCCcore-11.3.0 # for testing
# module avail
# module load FFmpeg/4.3.2-GCCcore-10.3.0 - this cannot be loaded
# which ffmpeg
module list
source activate speechbrain_venv

python _TEAM2_speechbrain_gpu_setup_test.py

pytest ../tests --device='cuda'