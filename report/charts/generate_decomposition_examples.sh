#! bin/bash

python ./report/charts/generate_decomposition_example.py --t-parameter 0.4
python ./report/charts/generate_decomposition_example.py --t-parameter 0.8

python ./report/charts/generate_decomposition_example.py --dataset wine
python ./report/charts/generate_decomposition_example.py --dataset breast_cancer
python ./report/charts/generate_decomposition_example.py --dataset digits
