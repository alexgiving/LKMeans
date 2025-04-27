#! bin/bash

LOGDIR=experiments_data/logs_super/detailed_self_supervised_clustering_real_data_1000
set -ex

mkdir -p ${LOGDIR}

# VALUES
DATASET=$1
MINKOSKI_VALUES=(2)
PREPROCESSOR_VALUES=(pca spectral_embeddings locally_linear_embeddings mds isomap umap)
PREPROCESS_COMPONENT=$2

# Constants
CLUSTERING=lkmeans
REPEATS=1000

for MINKOVSKI in "${MINKOSKI_VALUES[@]}";do
    for PREPROCESSOR in "${PREPROCESSOR_VALUES[@]}";do
        # for DATASET in "${DATASETS[@]}";do
            # for PREPROCESS_COMPONENT in "${PREPROCESS_COMPONENTS[@]}";do


NAME="${CLUSTERING}_|_self_supervision_${PREPROCESSOR}_|_self_supervision_n_components_${PREPROCESS_COMPONENT}_|_dataset_${DATASET}_|_minkowski_${MINKOVSKI}_|_repeats_${REPEATS}.log"

                echo ${LOGDIR}/${NAME}
                PARAMETERS="
                --dataset ${DATASET} \
                --minkowski-parameter ${MINKOVSKI} \
                --clustering-algorithm ${CLUSTERING} \
                --repeats ${REPEATS} \
                --self-supervised-preprocessor-algorithm ${PREPROCESSOR} \
                --self-supervised-components ${PREPROCESS_COMPONENT} \
                "
                python lkmeans/examples/main.py ${PARAMETERS} &> ${LOGDIR}/${NAME}
            # done
        # done
    done
done

echo DONE
