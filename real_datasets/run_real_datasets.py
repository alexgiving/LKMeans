'''Evaluation of LKMeans algorithm on real datasets'''

import keras.datasets
import sklearn.datasets

from real_datasets.utils import (make_experiment, preprocess_cifar,
                                 preprocess_mnist)


def main() -> None:
    # # Digits
    # print('='*50, '\nDigits dataset results')
    # data, labels = sklearn.datasets.load_digits(return_X_y=True)
    # make_experiment(data, labels)


    # Wine
    print('='*50, '\nWine dataset results')
    data, labels = sklearn.datasets.load_wine(return_X_y=True)
    make_experiment(data, labels)


    # Breast Cancer
    print('='*50, '\nBreast Cancer dataset results')
    data, labels = sklearn.datasets.load_breast_cancer(return_X_y=True)
    make_experiment(data, labels)


    # MNIST
    print('='*50, '\nMNIST dataset results')
    (data, labels), (_, _) = keras.datasets.mnist.load_data()
    data = preprocess_mnist(data)
    make_experiment(data, labels)


    # Fashion-MNIST
    print('='*50, '\nFashion-MNIST dataset results')
    (data, labels), (_, _) = keras.datasets.fashion_mnist.load_data()
    data = preprocess_mnist(data)
    make_experiment(data, labels)


    # # CIFAR10
    # print('='*50, '\nCIFAR10 dataset results')
    # (data, labels), (_, _) = keras.datasets.cifar10.load_data()
    # data = preprocess_cifar(data)
    # labels = labels.squeeze()
    # make_experiment(data, labels)


    # # CIFAR100
    # print('='*50, '\nCIFAR100 dataset results')
    # (data, labels), (_, _) = keras.datasets.cifar100.load_data()
    # data = preprocess_cifar(data)
    # labels = labels.squeeze()
    # make_experiment(data, labels)



if __name__ == '__main__':
    main()
