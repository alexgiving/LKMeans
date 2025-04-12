from typing import List, Union

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray
from sklearn import datasets

from lkmeans import LKMeans


def _get_test_p() -> list[float | int]:
    p = [0.2, 0.5, 1, 2]
    return p


def get_data() -> NDArray:
    num_objects = np.random.randint(5, 20)
    num_features = np.random.randint(1, 20)
    return np.array(np.random.uniform(-10, 10, size=(num_objects, num_features)))


@pytest.mark.api
@pytest.mark.parametrize("p", _get_test_p())
def test_general_processing(p: float | int) -> None:
    data = get_data()

    lkmeans = LKMeans(n_clusters=2, p=p)
    lkmeans.fit_predict(data)
    print("Inertia", lkmeans.inertia_)
    print("Centers", lkmeans.cluster_centers_)


def convert_from_ndarray(data: NDArray, data_type: str) -> Union[List, pd.DataFrame, pd.Series]:
    if data_type == "list":
        return list(data.tolist())
    if data_type == "frame":
        return pd.DataFrame(data.tolist())
    if data_type == "series":
        return pd.Series(data.tolist())
    raise ValueError("Unsupported conversion type")


@pytest.mark.api
@pytest.mark.parametrize("data_type", ["list", "frame"])
@pytest.mark.parametrize("p", _get_test_p())
def test_input_data_conversion(data_type: str, p: float | int) -> None:
    data = get_data()
    data = convert_from_ndarray(data, data_type)

    lkmeans = LKMeans(n_clusters=2, p=p)
    lkmeans.fit_predict(data)
    print("Inertia", lkmeans.inertia_)
    print("Centers", lkmeans.cluster_centers_)


pandas_dataset_loader_map = {
    "wine": datasets.load_wine,
    "diabetes": datasets.load_diabetes,
    "iris": datasets.load_iris,
}


@pytest.mark.api
@pytest.mark.parametrize("dataset_name", ["wine", "diabetes", "iris"])
def test_input_data_frame(dataset_name: str) -> None:
    data, targets = pandas_dataset_loader_map[dataset_name](as_frame=True, return_X_y=True)
    num_classes = len(set(targets))

    # Remove data to accelerate test
    num_test_samples = max(num_classes, 10)
    data = data[:num_test_samples]

    lkmeans = LKMeans(n_clusters=num_classes, p=2)
    lkmeans.fit_predict(data)
    print("Inertia", lkmeans.inertia_)
    print("Centers", lkmeans.cluster_centers_)
