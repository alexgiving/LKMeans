#! bin/bash

LOGDIR=experiments_data/logs_super/unsupervised_clustering_1000_100
set -ex

mkdir -p ${LOGDIR}

# VALUES
MINKOSKI_VALUES=(2)
T_VALUES=(-0.8 -0.6 -0.4 0 0.4 0.6 0.8)
N_POINTS_VALUES=(1000)
DIMENSION_VALUES=(20)
NUM_CLUSTERS_VALUES=(2)

# Constants
CLUSTERING=lkmeans
REPEATS=100


for NUM_CLUSTERS in "${NUM_CLUSTERS_VALUES[@]}";do
    for MINKOVSKI in "${MINKOSKI_VALUES[@]}";do
        for T in "${T_VALUES[@]}";do
            for N_POINTS in "${N_POINTS_VALUES[@]}";do
                for DIMENSION in "${DIMENSION_VALUES[@]}";do

NAME="${CLUSTERING}_|_num-clusters_${NUM_CLUSTERS}_|_minkowski_${MINKOVSKI}_|_t_${T}_|_n-points_${N_POINTS}_|_dimension_${DIMENSION}_|_repeats_${REPEATS}.log"

                    echo ${NAME}
                    PARAMETERS="
                    --num-clusters ${NUM_CLUSTERS} \
                    --minkowski-parameter ${MINKOVSKI} \
                    --t-parameter ${T} \
                    --n-points ${N_POINTS} \
                    --dimension ${DIMENSION} \
                    --clustering-algorithm ${CLUSTERING} \
                    --repeats ${REPEATS} \
                    "
                    python lkmeans/examples/main.py ${PARAMETERS} &> ${LOGDIR}/${NAME}
                done
            done
        done
    done
done

