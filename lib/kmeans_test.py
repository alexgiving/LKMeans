"""Test KMeans."""
import numpy as np
import pytest
from matplotlib import pyplot as plt

from lib.kmeans import KMeans


def generate_random_centers(start, stop, k):
    """Generate k random centers."""
    centers = []
    for _ in range(k):
        centers.append(np.random.randint(start, stop, 2))
    return centers


def generate_random_data_by_centers(n_dots, centers):
    """Generate data."""
    data = []
    for center in centers:
        new_sample = np.random.randn(n_dots,2) + center
        data.append(new_sample)
    return np.concatenate(data, axis = 0)


def plot(data, centers, name_postfix=""):
    """Plot data and centers."""
    plt.scatter(data[:,0], data[:,1], s=7)
    plt.scatter(centers[:,0], centers[:,1], marker='*', c='g', s=150)
    if name_postfix:
        plt.savefig(f'images/myfig_{name_postfix}.png')
    else:
        plt.savefig('images/myfig.png')
    plt.close()


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("parameter", [2, 0.1, 0.01])
def test_kmeans(k, parameter):
    """Test KMeans."""
    centres = generate_random_centers(0, 10, k)
    data = generate_random_data_by_centers(200, centres)

    kmeans = KMeans(k=k, parameter=parameter)
    centers = kmeans.fit(data)
    plot(data, centers, name_postfix=k)
