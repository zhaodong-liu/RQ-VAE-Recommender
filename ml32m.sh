#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --cpus-per-task=32
#SBATCH --mem=160GB
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:1 
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate TIGER
cd /scratch/zl4789/RQ-VAE-Recommender
python3 train_rqvae_ml32m.py configs/rqvae_ml32m.gin
python3 train_decoder_ml32m.py configs/decoder_ml32m.gin
conda deactivate