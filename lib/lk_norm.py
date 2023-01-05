import numpy as np

from lib.errors import InvalidDistanceMetricException


def lk_norm(x, y, p=2):
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
