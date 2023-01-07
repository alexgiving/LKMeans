"""Main module."""
import numpy as np
from sklearn.datasets import make_blobs

from lib.distance import lk_norm_matrix


def main():
    """Main function."""
    (vector_1, vector_2), *_ = make_blobs(n_samples=2, centers=2, random_state=42)

    vector = np.array([vector_1, vector_2])
    print(vector)
    res = lk_norm_matrix(vector, vector, parameter=0.001)
    print(res)

if __name__ == '__main__':
    main()
