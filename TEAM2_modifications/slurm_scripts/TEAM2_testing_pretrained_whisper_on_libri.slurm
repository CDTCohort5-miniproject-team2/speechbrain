#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --mem=32G
#SBATCH --output=testing_pretrained-%j.out

module load Anaconda3/2022.10
source activate speechbrain_venv

python ../_testing_pretrained_whisper_on_libri.py