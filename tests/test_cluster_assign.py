import numpy as np
import pytest

from lib.kmeans import assign_to_cluster
from lib.types import p_type


def _get_test_params():
    params = [
        (
            np.array([
                [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
            ), np.array([[0, 0], [1, 1]]),
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        ),
        (
            np.array([
                [0.500001, 0.50000001], [0, 0], [0, 0], [0, 0], [0, 0],
                [1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
            ), np.array([[0, 0], [1, 1]]),
            [1, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        )
    ]
    return params


def _get_test_p():
    p = [0.2, 0.5, 1, 2]
    return p


@pytest.mark.kmeans
@pytest.mark.parametrize("test_input, centroids, expected_output", _get_test_params())
@pytest.mark.parametrize("p", _get_test_p())
def test_assign_to_cluster(test_input, centroids, expected_output, p: p_type):
    _, label = assign_to_cluster(test_input, centroids, len(centroids), p)
    np.testing.assert_array_equal(expected_output, label)