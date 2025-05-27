#! bin/bash

LOGDIR=experiments_data/logs_super/unsupervised_clustering_real_data_1000
set -ex

mkdir -p ${LOGDIR}

# VALUES
MINKOSKI_VALUES=(2)

# Constants
CLUSTERING=lkmeans
REPEATS=100

DATASET=$1
# for DATASET in "${DATASETS[@]}";do
    for MINKOVSKI in "${MINKOSKI_VALUES[@]}";do

NAME="${CLUSTERING}_|_num-clusters_${NUM_CLUSTERS}_|_dataset_${DATASET}_|_minkowski_${MINKOVSKI}__|_repeats_${REPEATS}.log"

        echo ${NAME}
        PARAMETERS="
        --dataset ${DATASET} \
        --minkowski-parameter ${MINKOVSKI} \
        --clustering-algorithm ${CLUSTERING} \
        --repeats ${REPEATS} \
        "
        python lkmeans/examples/main.py ${PARAMETERS} &> ${LOGDIR}/${NAME}
    done
# done

