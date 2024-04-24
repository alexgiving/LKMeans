'''Evaluation of LKMeans algorithm on real datasets'''
from sklearn import datasets

from lkmeans.examples.utils import make_experiment


def main() -> None:
    # Wine
    print('='*50, '\nWine dataset results')
    data, labels = datasets.load_wine(return_X_y=True)
    make_experiment(data, labels)


    # Breast Cancer
    print('='*50, '\nBreast Cancer dataset results')
    data, labels = datasets.load_breast_cancer(return_X_y=True)
    make_experiment(data, labels)


if __name__ == '__main__':
    main()
