import numpy as np
from numpy.typing import NDArray

from lkmeans.optimizers import mean_optimizer, median_optimizer, segment_slsqp_optimizer


def get_test_data(size: int) -> tuple[NDArray, float]:
    centre = +89.9573

    data = np.random.random(size)
    reverted_data = data * -1
    samples = np.concatenate([data, reverted_data])
    samples = samples + centre
    return samples, centre


def main() -> None:
    samples, centre = get_test_data(50)

    print(f'Expected centre: {centre :.5f}')

    print(f'Optimizer median: {median_optimizer(samples) :.5f}')
    print(f'Optimizer mean: {mean_optimizer(samples) :.5f}')

    for p in [0.2, 0.5]:
        print(
            f'Optimizer SLSQP (p={p}): {segment_slsqp_optimizer(samples, p)}')


if __name__ == '__main__':
    main()
