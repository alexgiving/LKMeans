DATASETS=(wine breast_cancer iris digits mnist cifar10)
PREPROCESS_COMPONENTS=(19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2)
PREPROCESSOR_VALUES=(pca spectral_embeddings locally_linear_embeddings mds isomap umap)

for DATASET in "${DATASETS[@]}";do
    for PREPROCESS_COMPONENT in "${PREPROCESS_COMPONENTS[@]}";do
        for PREPROCESSOR in "${PREPROCESSOR_VALUES[@]}";do
            sbatch ./lkmeans/examples/scripts/sbatch_self_supervised_real_data.sh ${DATASET} ${PREPROCESS_COMPONENT} ${PREPROCESSOR}
        done
    done
done


# for DATASET in "${DATASETS[@]}";do
#     sbatch ./lkmeans/examples/scripts/sbatch_unsupervised_real_data.sh ${DATASET}
# done
