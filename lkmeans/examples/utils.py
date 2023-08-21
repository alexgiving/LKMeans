import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans

from lkmeans import LKMeans


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def preprocess_cifar(data: np.ndarray) -> np.ndarray:
    data = rgb2gray(data)
    data = data.astype('float32')
    data = data.reshape(len(data),-1)
    return data


def preprocess_mnist(data: np.ndarray) -> np.ndarray:
    data = data.astype('float32')
    data = data/255.0
    data = data.reshape(len(data),-1)
    return data


def get_metrics(true_labels, predicted_labels, inertia: float):
    ami = metrics.adjusted_mutual_info_score(true_labels, predicted_labels)
    ari = metrics.adjusted_rand_score(true_labels, predicted_labels)
    return {'ARI': ari, 'AMI': ami, 'Inertia': inertia}


def make_experiment(data, labels):
    (n_samples, n_features), n_digits = data.shape, np.unique(labels).size
    print(f"# classes: {n_digits}; # samples: {n_samples}; # features {n_features}")


    kmeans = KMeans(init='random', n_clusters=n_digits, n_init=1, random_state=0)
    result_labels = kmeans.fit_predict(data)
    print('k-Means (sklearn)', get_metrics(labels, result_labels, kmeans.inertia_))


    for p in [2, 5, 1, 0.5]:
        lkmeans = LKMeans(n_clusters=n_digits, p=p)
        result_labels = lkmeans.fit_predict(data)
        print(f'LKMeans (p={p})', get_metrics(labels, result_labels, lkmeans.inertia_))
