import numpy as np
from sklearn.datasets import make_blobs

from lib.distance import lk_norm_XY, lk_norm


def main():

    (X, Y), labels = make_blobs(n_samples=2, centers=2, random_state=42)
    # print(X, Y)

    # res = lk_norm_XY(X, Y, p=0.1)
    # print(res)

    XY = np.array([X, Y])
    print(XY)
    res = lk_norm(XY, XY, p=0.001)
    print(res)

if __name__ == '__main__':
    main()
