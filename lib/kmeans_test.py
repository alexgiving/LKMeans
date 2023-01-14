"""Test KMeans."""
import time
from pathlib import Path

import imageio
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


def plot_colored(data, centers, labels, name_postfix="", title=""):
    """Plot data and centers."""
    sns.scatterplot(x=data[:,0],
                    y=data[:,1],
                    hue=labels,
                    palette="deep",
                    legend=None
                    )
    plt.plot(centers[:,0], centers[:,1], 'k+', markersize=10)
    if title:
        plt.title(title)
    if name_postfix:
        plt.savefig(f'images/myfig_{name_postfix}.png')
    else:
        plt.savefig('images/myfig.png')
    plt.close()


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("parameters", [[0.01, 0.1, 2]])
def test_kmeans(k, parameters):
    """Test KMeans."""
    data, *_ = make_blobs(n_samples=200, centers=k, n_features=768)

    for parameter in parameters:
        kmeans = KMeans(k=k, parameter=parameter)
        _ = kmeans.fit(data)


@pytest.mark.parametrize("k", [1, 5, 10])
@pytest.mark.parametrize("parameters", [[0.01, 0.1, 2]])
def test_fit_transform_kmeans(k, parameters):
    """
    Provides the algorithm work status.
    API same as skitlearn
    """
    data, *_ = make_blobs(n_samples=200, centers=k, n_features=768)

    for parameter in parameters:
        kmeans = KMeans(k=k, parameter=parameter)
        centers = kmeans.fit(data)
        _ = kmeans.transform(centers, data)
        _ = kmeans.fit_transform(data)


def make_gif():
    """Make gif."""
    k = 4
    data, labels, *_ = make_blobs(n_samples=200, centers=k)

    for parameter in [2, 0.1, 0.01]:
        start_time = time.time()
        kmeans = KMeans(k=k, parameter=parameter)
        centers = kmeans.fit(data)
        print(f"({k}, {parameter}) - Time: {time.time() - start_time}")
        plot_colored(data, centers, labels, name_postfix=f"{k}_{parameter}",
                    title=f"KMeans with L_{parameter} norm")

    frames_per_sec = 0.5
    path = Path('images')
    image_paths = path.glob('*.png')
    gif_name = Path("cache") / f"gif_{k}_{frames_per_sec}.gif"

    index = 0
    while gif_name.exists():
        index += 1
        gif_name = Path("cache") / f"gif_{k}_{frames_per_sec}_v{index}.gif"
    frames = [imageio.imread(filename) for filename in image_paths]
    imageio.mimsave(gif_name, frames, format='GIF', fps=frames_per_sec)
    # Add coloring clusters, not original data points


def test_main():
    """Test KMeans."""
    k = 3
    # 768 - embedding of BERT model
    data, *_ = make_blobs(n_samples=1000, centers=k, n_features=768)

    for parameter in [0.01]:
        kmeans = KMeans(k=k, parameter=parameter)
        centers = kmeans.fit(data)
        clusters = kmeans.transform(centers, data)
        print(clusters)

        clusters = kmeans.fit_transform(data)
        print(clusters)

if __name__ == "__main__":
    test_main()
