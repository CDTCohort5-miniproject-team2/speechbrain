#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=80G

module load Anaconda3/2022.10
source activate speechbrain_venv

python ../recipes/LibriSpeech/ASR/transformer/TEAM2_train_with_whisper.py ../recipes/LibriSpeech/ASR/transformer/hparams/TEAM2_train_hf_whisper.yaml