#!/bin/bash
#SBATCH --output=jobs/Job.%j.out
#SBATCH --error=jobs/Job.%j.err
#SBATCH --partition=rtx8000,v100
#SBATCH --cpus-per-task=10
#SBATCH --mem=32GB
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:1      
#SBATCH --mail-type=ALL          
#SBATCH --mail-user=zl4789@nyu.edu
#SBATCH --requeue

source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate TIGER
cd /scratch/zl4789/RQ-VAE-Recommender
python3 train_decoder.py configs/decoder_amazon.gin
conda deactivate