import pytest
from lib.distance import lk_norm, get_lambda_minkowski
import numpy as np
import scipy


@pytest.mark.parametrize("XY, p", [
    (np.array([[-2.743351, 8.78014917], [ 6.21909165, 2.74060441]]), 0.001),
    (np.array([[-8.743351, 0.4917], [ -4.9165, 15.]]), 0.01),
    ])
def test_hello(XY, p):
    res_minkowski = lk_norm(XY, XY, p)

    metric = get_lambda_minkowski(p)
    res_custom = scipy.spatial.distance.cdist(XY, XY, metric)
    
    np.testing.assert_array_equal(res_minkowski, res_custom)
