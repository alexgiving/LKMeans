from typing import List, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from lkmeans.clustering.base import assign_to_cluster
from lkmeans.clustering.semi_supervised.utils import (
    assign_to_cluster_with_supervision,
    select_supervisor_targets,
)
from lkmeans.distance import DistanceCalculator


def _get_test_params() -> List[Tuple[NDArray, NDArray, NDArray]]:
    params = [
        (
            np.array(
                [
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ]
            ),
            np.array([[0, 0], [1, 1]]),
            np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        ),
        (
            np.array(
                [
                    [0.500001, 0.50000001],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [0, 0],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                    [1, 1],
                ]
            ),
            np.array([[0, 0], [1, 1]]),
            np.array([1, 0, 0, 0, 0, 1, 1, 1, 1, 1]),
        ),
    ]
    return params


def _get_test_p() -> list[float | int]:
    p = [0.2, 0.5, 1, 2]
    return p


def _get_test_supervision_ratio() -> list[float]:
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    return ratios


@pytest.mark.lkmeans
@pytest.mark.parametrize("test_input, centroids, expected_output", _get_test_params())
@pytest.mark.parametrize("p", _get_test_p())
def test_assign_to_cluster(
    test_input: NDArray, centroids: NDArray, expected_output: NDArray, p: float | int
) -> None:
    distance_calculator = DistanceCalculator(p)
    _, label = assign_to_cluster(
        test_input, centroids, len(np.unique(centroids)), distance_calculator
    )
    np.testing.assert_array_equal(expected_output, label)


@pytest.mark.lkmeans
@pytest.mark.parametrize("test_input, centroids, expected_output", _get_test_params())
@pytest.mark.parametrize("p", _get_test_p())
@pytest.mark.parametrize("supervision_ratio", _get_test_supervision_ratio())
def test_assign_to_cluster_with_supervision(
    test_input: NDArray,
    centroids: NDArray,
    expected_output: NDArray,
    p: float | int,
    supervision_ratio: float,
) -> None:
    distance_calculator = DistanceCalculator(p)
    supervision_targets = select_supervisor_targets(expected_output, supervision_ratio)
    _, label = assign_to_cluster_with_supervision(
        test_input,
        centroids,
        len(np.unique(centroids)),
        distance_calculator,
        supervision_targets,
    )
    np.testing.assert_array_equal(expected_output, label)
