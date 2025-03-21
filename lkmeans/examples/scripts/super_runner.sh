T_VALUES=(-0.8 -0.6 -0.4 0.6 0.8) # -0.8 -0.6 -0.4 0 0.4 0.6 0.8
PREPROCESS_COMPONENTS=(19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2)

for T in "${T_VALUES[@]}";do
    for PREPROCESS_COMPONENT in "${PREPROCESS_COMPONENTS[@]}";do
        sbatch -A proj_1538 ./lkmeans/examples/scripts/sbatch_self_supervised.sh ${T} ${PREPROCESS_COMPONENT}
    done
done
