import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from lkmeans import LKMeans

p_values = [0.5, 1, 2, 5]


def get_data() -> NDArray:
    return np.random.uniform(-10,10, size=(100, 50))


@pytest.mark.api
@pytest.mark.parametrize('p', p_values)
def test_general_processing(p: float | int) -> None:
    data = get_data()

    lkmeans = LKMeans(n_clusters=2, p=p)
    lkmeans.fit_predict(data)
    print('Inertia', lkmeans.inertia_)
    print('Centers', lkmeans.cluster_centers_)


def convert_from_ndarray(data: NDArray, data_type: str) -> list | pd.DataFrame | pd.Series:
    if data_type == 'list':
        return data.tolist()
    if data_type == 'frame':
        return pd.DataFrame(data.tolist())
    if data_type == 'series':
        return pd.Series(data.tolist())
    raise ValueError('Unsupported conversion type')


@pytest.mark.api
@pytest.mark.parametrize('data_type', ['list', 'frame', 'series'])
def test_input_data_conversion(data_type: str) -> None:
    data = get_data()
    data = convert_from_ndarray(data, data_type)

    lkmeans = LKMeans(n_clusters=2, p=2)
    lkmeans.fit_predict(data)
    print('Inertia', lkmeans.inertia_)
    print('Centers', lkmeans.cluster_centers_)
