#!/bin/bash
#SBATCH --partition=dcs-gpu-test
#SBATCH --qos=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --job-name=train_whisper
#SBATCH --output=train_whisper.%j.test.out
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --account=dcs-res

module load Anaconda3/5.3.0

source activate speechbrain

module load FFmpeg/4.2.2-GCCcore-9.3.0

module load libsndfile/1.0.28-GCCcore-9.3.0

pip install transformers

srun --export=ALL python train_with_whisper.py hparams/train_hf_whisper_encoder.yaml --skip_prep=True

