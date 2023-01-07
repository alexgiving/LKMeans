"""Module providing Distance metric functions."""
import numpy as np
import scipy

from lib.deprecated import deprecated
from lib.errors import InvalidDistanceMetricException


@deprecated
def lk_norm_vector(vector_1, vector_2, parameter=2):
    """Lk norm of two vectors vector_1 and vector_2.

    Parameters
    ----------
    vector_1 : array-like
        First vector.
    vector_2 : array-like
        Second vector.
    parameter : float
        The order of the norm. Default is 2 'Euclidean'.

    Returns
    -------
    dist : float
        Distance between the input arrays.

    Examples
    --------
    >>> lk_norm([10, 1, 2], [3, 4, 5], 0.1)
    236858.47165907384

    """
    vector_1 = np.asarray(vector_1)
    vector_2 = np.asarray(vector_2)
    if parameter == 0:
        raise InvalidDistanceMetricException("parameter cannot be 0.")
    dist = np.sum(np.abs(vector_1-vector_2)**parameter)**(1/parameter)
    return dist


def get_lambda_minkowski(parameter):
    """Get lambda function for Minkowski distance."""
    return lambda u, v: np.sum(np.abs(u-v)**parameter)**(1/parameter)


def lk_norm_matrix(vector_1, vector_2, parameter=2):
    """Lk norm of two vectors x and y.

    Parameters
    ----------
    vector_1 : array-like
        First vector.
    vector_2 : array-like
        Second vector.
    parameter : float
        The order of the norm. Default is 2 'Euclidean'.

    Returns
    -------
    dist : matrix of floats
        Distance between each point of input arrays.
    """
    vector_1 = np.asarray(vector_1)
    vector_2 = np.asarray(vector_2)

    dist = scipy.spatial.distance.cdist(vector_1, vector_2, 'minkowski', p=parameter)
    return dist
