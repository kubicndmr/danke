#!/bin/bash -l
#SBATCH --job-name=SDG
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a40:1
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err
##SBATCH --mail-type=end,fail
#SBATCH --time=23:59:00
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

source ~/.bashrc
echo "Your job is running on" $(hostname)

# Set proxy to access internet from the node -C a100_80
export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

# Python
module load python
echo "module load python"

# Conda
source activate kenv
echo "source activate kenv"

# Run training script
#python sdg.py -t SynPoCaP-15 -n 35
for ((i=1; i<=4; i++)); do
    python spr.py --real_dataset_size 0 --syn_dataset_size 50 --real_data_path TextualSPR-Dataset/PoCaP-large-v3 --syn_data_path TextualSPR-Dataset/SynPoCaP
done