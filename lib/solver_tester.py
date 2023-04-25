import numpy as np

from lib.optimizers import (mean_optimizer, median_optimizer,
                            segment_SLSQP_optimizer)


def get_test_data(size: int) -> np.ndarray:
    data = np.random.random(size) + 0.1
    reverted_data = data * -1
    samples = np.concatenate([data, reverted_data])
    samples = samples + 0.1
    return samples


def main() -> None:
    samples = get_test_data(50)

    print(np.median(samples))
    print(median_optimizer(samples))
    print(mean_optimizer(samples))
    print(segment_SLSQP_optimizer(samples, 0.8))


if __name__ == '__main__':
    main()
