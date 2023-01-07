"""Test KMeans."""
import numpy as np
import pytest
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs

from lib.deprecated import deprecated
from lib.kmeans import KMeans


@deprecated
def generate_random_centers(start, stop, k):
    """Generate k random centers."""
    centers = []
    for _ in range(k):
        centers.append(np.random.randint(start, stop, 2))
    return centers


@deprecated
def generate_random_data_by_centers(n_dots, centers):
    """Generate data."""
    data = []
    for center in centers:
        new_sample = np.random.randn(n_dots,2) + center
        data.append(new_sample)
    return np.concatenate(data, axis = 0)

@deprecated
def plot(data, centers, name_postfix=""):
    """Plot data and centers."""
    plt.scatter(data[:,0], data[:,1], s=7)
    plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
    if name_postfix:
        plt.savefig(f'images/myfig_{name_postfix}.png')
    else:
        plt.savefig('images/myfig.png')
    plt.close()


def plot_colored(data, centers, labels, name_postfix=""):
    """Plot data and centers."""
    sns.scatterplot(x=data[:,0],
                    y=data[:,1],
                    hue=labels,
                    palette="deep",
                    legend=None
                    )
    plt.plot(centers[:,0], centers[:,1], 'k+', markersize=10)
    if name_postfix:
        plt.savefig(f'images/myfig_{name_postfix}.png')
    else:
        plt.savefig('images/myfig.png')
    plt.close()


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("parameters", [[0.01, 0.1, 2]])
def test_kmeans(k, parameters):
    """Test KMeans."""
    data, labels, *_ = make_blobs(n_samples=200, centers=k)

    for parameter in parameters:
        kmeans = KMeans(k=k, parameter=parameter)
        centers = kmeans.fit(data)
        plot_colored(data, centers, labels, name_postfix=f"{k}_{parameter}")
        print('creating gif\n')
