import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from lkmeans.distance import DistanceCalculator


def median_optimizer(dimension_slice: NDArray) -> float:
    '''
    Standard K-Medoids optimizer.
    Returns the optimal `median` point of input array
    '''
    return float(np.median(dimension_slice))


def mean_optimizer(dimension_slice: NDArray) -> float:
    '''
    Standard K-Means optimizer.
    Returns the optimal `mean` point of input array
    '''
    return float(np.mean(dimension_slice))


def bound_optimizer(dimension_slice: NDArray, p: float | int) -> float:
    '''
    Special LKMeans optimizer.
    Based on idea that for 0 < p < 1 the minkowski function is a concave function.
    Returns the optimal point of input array with Minkowski metric parameter `p`
    '''
    points = np.unique(dimension_slice)
    distance_calculator = DistanceCalculator(p)

    optimal_point = points[0]
    smallest_distant = distance_calculator.get_distance(optimal_point, dimension_slice)
    for applicant in points:
        distance_of_applicant = distance_calculator.get_distance(applicant, dimension_slice)
        if distance_of_applicant < smallest_distant:
            optimal_point = applicant
            smallest_distant = distance_of_applicant
    return float(optimal_point)


def parallel_segment_slsqp_optimizer(cluster: NDArray, dim: int, p: float | int):
    optimize_slice = partial(segment_slsqp_optimizer, p=p)
    dimension_slices = [cluster[:, coordinate_id] for coordinate_id in range(dim)]
    with Pool(cpu_count()) as pool:
        new_centroid = pool.map(optimize_slice, dimension_slices)
    return new_centroid


def segment_slsqp_optimizer(dimension_slice: NDArray, p: float | int, tol: float = 1e-1_000) -> float:
    dimension_slice = np.unique(dimension_slice)
    distance_calculator = DistanceCalculator(p)

    median = np.median(dimension_slice)
    fun_median = distance_calculator.get_distance(
        np.array(median), dimension_slice)

    minimized_fun_median = fun_median
    for bound_id in range(len(dimension_slice) - 1):

        bounds = [(dimension_slice[bound_id],
                   dimension_slice[bound_id + 1])]

        x0 = np.mean(bounds)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(
                fun=lambda centre: distance_calculator.get_distance(
                    centre, dimension_slice),
                x0=x0,
                method='SLSQP',
                bounds=bounds,
                tol=tol
            )

        if res.success:
            minima_point = res.x[0]
            minimal_point_value = res.fun
            if minimal_point_value < minimized_fun_median:
                minimized_fun_median = minimal_point_value
                median = minima_point
    return float(median)


def slsqp_optimizer(dimension_slice: NDArray, p: float | int, tol: float = 1e-1_000):
    distance_calculator = DistanceCalculator(p)

    x0 = np.mean(dimension_slice)
    bounds = [(min(dimension_slice), max(dimension_slice))]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = minimize(
            fun=lambda centre: distance_calculator.get_distance(
                centre, dimension_slice),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            tol=tol
        ).x[0]
    return float(res)
