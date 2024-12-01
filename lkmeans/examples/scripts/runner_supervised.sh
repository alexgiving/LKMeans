#! bin/bash

LOGDIR=logs
set -ex

mkdir -p ${LOGDIR}

# VALUES
MINKOSKI_VALUES=(0.5 1 2 5)
T_VALUES=(0 0.2 0.4 0.6 0.8)
N_POINTS_VALUES=(100 500 1000)
CLUSTERINGS_VALUES=(soft_semi_supervised_lkmeans hard_semi_supervised_lkmeans)
SUPERVISION_RATIO_VALUES=(0.1 0.15 0.2)
DIMENSION_VALUES=(20)
NUM_CLUSTERS_VALUES=(2 3)

# Constants
REPEATS=10

source .env

for NUM_CLUSTERS in "${NUM_CLUSTERS_VALUES[@]}";do
    for MINKOVSKI in "${MINKOSKI_VALUES[@]}";do
        for T in "${T_VALUES[@]}";do
            for N_POINTS in "${N_POINTS_VALUES[@]}";do
                for DIMENSION in "${DIMENSION_VALUES[@]}";do
                    for CLUSTERING in "${CLUSTERINGS_VALUES[@]}";do
                        for SUPERVISION_RATIO in "${SUPERVISION_RATIO_VALUES[@]}";do


NAME="${CLUSTERING}_|_supervision_${SUPERVISION_RATIO}_|_num-clusters_${NUM_CLUSTERS}_|_minkowski_${MINKOVSKI}_|_t_${T}_|_n-points_${N_POINTS}_|_dimension_${DIMENSION}_|_repeats_${REPEATS}.log"

                            echo ${NAME}
                            PARAMETERS="
                            --num-clusters ${NUM_CLUSTERS} \
                            --minkowski-parameter ${MINKOVSKI} \
                            --t-parameter ${T} \
                            --n-points ${N_POINTS} \
                            --dimension ${DIMENSION} \
                            --clustering-algorithm ${CLUSTERING} \
                            --supervision-ratio ${SUPERVISION_RATIO} \
                            --repeats ${REPEATS} \
                            "
                            python lkmeans/examples/main.py ${PARAMETERS} &> ${LOGDIR}/${NAME}
                        done
                    done
                done
            done
        done
    done
done

