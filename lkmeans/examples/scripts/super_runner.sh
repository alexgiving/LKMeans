T_VALUES=(-0.4) #-1 -0.5 0 0.4 0.6 0.8
PREPROCESS_COMPONENTS=(2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19)

for T in "${T_VALUES[@]}";do
    for PREPROCESS_COMPONENT in "${PREPROCESS_COMPONENTS[@]}";do
        sbatch ./lkmeans/examples/scripts/sbatch_self_supervised.sh ${T} ${PREPROCESS_COMPONENT}
    done
done
