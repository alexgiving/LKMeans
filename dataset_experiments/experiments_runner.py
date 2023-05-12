'''Evaluation of LKMeans algorithm on real datasets'''

import sklearn.datasets
from keras.datasets import mnist

from dataset_experiments.utils import make_experiment, preprocess_mnist

## MNIST
print('='*50, '\nMnist dataset results')
(data, label), (_, _) = mnist.load_data()
data = preprocess_mnist(data)
make_experiment(data, label)


## Digits
print('='*50, '\nDigits dataset results')
data, labels = sklearn.datasets.load_digits(return_X_y=True)
make_experiment(data, labels)


## Wine
print('='*50, '\nWine dataset results')
data, labels = sklearn.datasets.load_wine(return_X_y=True)
make_experiment(data, labels)


## Breast Cancer
print('='*50, '\nBreast Cancer dataset results')
data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
make_experiment(data, labels)

