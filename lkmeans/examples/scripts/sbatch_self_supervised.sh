#! /bin/bash
#SBATCH --job-name="self_supervised_clustering"
#SBATCH --output=%x_%j.out

#SBATCH --gpus=0
#SBATCH --time=0-2:0

module purge
module load Python

source deactivate
source activate lkmeans_venv

# Executable
export PYTHONPATH=${PYTHONPATH}:$(pwd)

srun bash ./lkmeans/examples/scripts/runner_self_supervised.sh $1 $2
