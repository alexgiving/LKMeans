import numpy as np
import scipy

from lib.deprecated import deprecated
from lib.errors import InvalidDistanceMetricException


@deprecated
def lk_norm_XY(x, y, p=2):
    """Lk norm of two vectors x and y.
    
    Parameters
    ----------
    x : array-like
        First vector.
    y : array-like
        Second vector.
    p : float
        The order of the norm. Default is 2 'Euclidean'.
    
    Returns
    -------
    dist : float
        Distance between the input arrays.
        
    Examples
    --------
    >>> lk_norm([10, 1, 2], [3, 4, 5], p=0.1)
    236858.47165907384
    
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == 0:
        raise InvalidDistanceMetricException("p cannot be 0.")
    dist = np.sum(np.abs(x-y)**p)**(1/p)
    return dist


def get_lambda_minkowski(p):
    metric = lambda u, v: np.sum(np.abs(u-v)**p)**(1/p)
    return metric


def lk_norm_test(x, y, p=2):
    x = np.asarray(x)
    y = np.asarray(y)
    l_p_norm = get_lambda_minkowski(p)

    import time
    N = 10000
    
    first = []
    for _ in range(N):
        st = time.time()
        dist_1 = scipy.spatial.distance.cdist(x, y, 'minkowski', p=p)
        et = time.time()
        first.append(et-st)
    print(f"Elapsed time: {np.mean(first)} seconds")
    

    sec = []
    for _ in range(N):
        st = time.time()
        dist_2 = scipy.spatial.distance.cdist(x, y, l_p_norm)
        et = time.time()
        sec.append(et-st)
    print(f"Elapsed time: {np.mean(sec)} seconds")

    print(f"Custom is faster?: {np.mean(sec) < np.mean(first)}")

    return dist_1==dist_2


def lk_norm(x, y, p=2):
    x = np.asarray(x)
    y = np.asarray(y)

    dist = scipy.spatial.distance.cdist(x, y, 'minkowski', p=p)
    return dist

