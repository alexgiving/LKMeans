import numpy as np
from sklearn.datasets import make_blobs

from lib.lk_norm import lk_norm


def main():

    (X, Y), labels = make_blobs(n_samples=2, centers=2, random_state=42)

    res = lk_norm(X, Y, p=0.1)
    print(res)

if __name__ == '__main__':
    main()
