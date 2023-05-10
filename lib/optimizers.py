import warnings
from functools import partial
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.optimize import minimize

from lib.minkowski import minkowski_distance


def median_optimizer(dimension_slice: np.ndarray) -> float:
    '''
    Standard KMeans optimizer.
    '''
    return float(np.median(dimension_slice))


def mean_optimizer(dimension_slice: np.ndarray) -> float:
    return float(np.mean(dimension_slice))


def bound_optimizer(dimension_slice: np.ndarray, p: float | int) -> float:
    '''
    Based on idea that for 0 < p < 1 the minkowski function is a concave function.
    '''
    points = np.unique(dimension_slice)

    result = points[0]
    f_result = minkowski_distance(result, dimension_slice, p)
    for pretendent in points:
        f_pretendent = minkowski_distance(pretendent, dimension_slice, p)
        if f_pretendent < f_result:
            result = pretendent
            f_result = f_pretendent
    return float(result)


def parallel_segment_slsqp_optimizer(cluster: np.ndarray, dim: int, p: float | int):
    optimize_slice = partial(segment_slsqp_optimizer, p=p)
    dimension_slices = [cluster[:, coordinate_id] for coordinate_id in range(dim)]
    with Pool(cpu_count()) as pool:
        new_centroid = pool.map(optimize_slice, dimension_slices)
    return new_centroid


def segment_slsqp_optimizer(dimension_slice: np.ndarray, p: float | int, tol: float = 1e-1_000) -> float:
    dimension_slice = np.unique(dimension_slice)

    median = np.median(dimension_slice)
    fun_median = minkowski_distance(
        np.array(median), dimension_slice, p)

    minimized_fun_median = fun_median
    for bound_id in range(len(dimension_slice) - 1):

        bounds = [(dimension_slice[bound_id],
                   dimension_slice[bound_id + 1])]

        x0 = np.mean(bounds)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            res = minimize(
                fun=lambda centre: minkowski_distance(
                    centre, dimension_slice, p),
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


def slsqp_optimizer(dimension_slice: np.ndarray, p: float | int, tol: float = 1e-1_000):
    x0 = np.mean(dimension_slice)
    bounds = [(min(dimension_slice), max(dimension_slice))]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        res = minimize(
            fun=lambda centre: minkowski_distance(
                centre, dimension_slice, p),
            x0=x0,
            method='SLSQP',
            bounds=bounds,
            tol=tol
        ).x[0]
    return float(res)
