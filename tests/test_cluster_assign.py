import numpy as np
import pytest

from lib.kmeans import assign_to_cluster
from lib.types import p_type

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

p = [0.001, 0.1, 0.2, 0.5, 0.9, 1, 2]


@pytest.mark.kmeans
@pytest.mark.parametrize("test_input, centroids, expected_output", params)
@pytest.mark.parametrize("p", p)
def test_assign_to_cluster(test_input, centroids, expected_output, p: p_type):
    _, label = assign_to_cluster(test_input, centroids, len(centroids), p)
    np.testing.assert_array_equal(expected_output, label)
