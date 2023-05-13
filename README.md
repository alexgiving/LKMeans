[![CI](https://github.com/alexgiving/LKMeans/actions/workflows/test.yml/badge.svg)](https://github.com/alexgiving/LKMeans/actions/workflows/test.yml)
# LKMeans

LKMeans is a Python library that extends the functionality of the KMeans algorithm by allowing clusterization using the Minkowski metric instead of the traditional Euclidean distance. The Minkowski metric provides better quality performance in certain scenarios, making LKMeans a valuable tool for clustering analysis.

## Features

- Clusterization using the Minkowski metric
- Improved quality performance compared to traditional KMeans
- Easy integration into existing machine learning pipelines

## Installation

You can install LKMeans:
```bash
export PYTHONPATH=${PYTHONPATH}:$(pwd)
```

## Usage
Using LKMeans is straightforward. Here's a simple example that demonstrates how to use the library:
```python
import numpy as np

from lib.lkmeans import LKMeans

# Generate some random data
X = np.random.rand(100, 2)

# Create an instance of LKMeans with the desired number of clusters and Minkowski parameter
lkmeans = LKMeans(n_clusters=3, p=0.8)

# Fit the model to the data
labels = lkmeans.fit_predict(X)
centers = lkmeans.cluster_centers_
inertia = lkmeans.inertia_

```
In this example, we first import the LKMeans class from the LKMeans library. We create an instance of LKMeans with the desired number of clusters (n_clusters) and the desired Minkowski distance parameter (p), and fit the model to the data using the fit method.


## Contributing
Contributions to LKMeans are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request on the GitHub repository.

## License
LKMeans is licensed under the MIT License. See the [LICENSE](./LICENSE) file for more information.

## Contact
For any questions or inquiries, please contact alexgiving@mail.ru.

Enjoy clustering with LKMeans!
