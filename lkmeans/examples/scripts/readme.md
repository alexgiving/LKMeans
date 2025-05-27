# Instruction how to run benchmarking


## Unsupervised clustering

### Generated data
1. Run benchmarking
    ```bash
    bash ./lkmeans/examples/scripts/runner_unsupervised.sh
    ```

1. Make SLURM job for benchmarking
    ```bash
    sbatch ./lkmeans/examples/scripts/sbatch_unsupervised.sh
    ```

## Semi-Supervised clustering

### Generated data
1. Run benchmarking
    ```bash
    bash ./lkmeans/examples/scripts/runner_semi_supervised.sh
    ```

1. Make SLURM job for benchmarking
    ```bash
    sbatch ./lkmeans/examples/scripts/sbatch_semi_supervised.sh
    ```

## Self-Supervised clustering

### Generated data
1. Run benchmarking
    ```bash
    bash ./lkmeans/examples/scripts/runner_self_supervised.sh
    ```

1. Make SLURM job for benchmarking
    ```bash
    sbatch ./lkmeans/examples/scripts/sbatch_self_supervised.sh
    ```

### Real data
1. Run benchmarking
    ```bash
    bash ./lkmeans/examples/scripts/runner_self_supervised_real_data.sh
    ```

1. Make SLURM job for benchmarking
    ```bash
    sbatch ./lkmeans/examples/scripts/sbatch_self_supervised_real_data.sh
    ```
