T_VALUES=(-0.8 -0.6 -0.4 0 0.4 0.6 0.8) # -0.8 -0.6 -0.4 0 0.4 0.6 0.8
PREPROCESS_COMPONENTS=(10)

for T in "${T_VALUES[@]}";do
    for PREPROCESS_COMPONENT in "${PREPROCESS_COMPONENTS[@]}";do
        sbatch ./lkmeans/examples/scripts/sbatch_self_supervised.sh ${T} ${PREPROCESS_COMPONENT}
    done
done
