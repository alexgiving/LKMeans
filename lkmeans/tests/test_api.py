import numpy as np
import pytest

from lkmeans import LKMeans

p_values = [0.5, 1, 2, 5]

@pytest.mark.api
@pytest.mark.parametrize("p", p_values)
def test_segment_slsqp_calculation(p) -> None:
    data = np.random.uniform(-10,10, size=(100, 50))

    lkmeans = LKMeans(n_clusters=2, p=p)
    lkmeans.fit_predict(data)
    print('Inertia', lkmeans.inertia_)
    print('Centers', lkmeans.cluster_centers_)
