DATASETS=(wine breast_cancer iris digits mnist cifar10)
PREPROCESS_COMPONENTS=(19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2)

for DATASET in "${DATASETS[@]}";do
    for PREPROCESS_COMPONENT in "${PREPROCESS_COMPONENTS[@]}";do
        sbatch ./lkmeans/examples/scripts/sbatch_self_supervised_real_data.sh ${DATASET} ${PREPROCESS_COMPONENT}
    done
done
