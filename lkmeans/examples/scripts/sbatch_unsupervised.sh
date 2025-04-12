#! /bin/bash
#SBATCH --job-name="unsupervised_clustering"
#SBATCH --output=%x_%j.out

#SBATCH --gpus=0
#SBATCH --time=1-0:0

module purge
module load Python

source deactivate
source activate lkmeans_venv

# Executable
export PYTHONPATH=${PYTHONPATH}:$(pwd)

srun bash ./lkmeans/examples/scripts/runner_unsupervised.sh
