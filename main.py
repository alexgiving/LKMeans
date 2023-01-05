import numpy as np

from lib.lk_norm import lk_norm


def main():

    X = np.random.random_sample(10)
    Y = np.random.random_sample(10)

    res = lk_norm(X, Y, p=0.1)
    print(res)

if __name__ == '__main__':
    main()
