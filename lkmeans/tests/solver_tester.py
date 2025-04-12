import numpy as np
from numpy.typing import NDArray

from lkmeans.optimizers import (
    BoundOptimizer,
    MeanOptimizer,
    MedianOptimizer,
    SegmentSLSQPOptimizer,
    SLSQPOptimizer,
)


def get_test_data(size: int) -> tuple[NDArray, float]:
    centre = +89.9573

    data = np.random.random(size)
    reverted_data = data * -1
    samples = np.concatenate([data, reverted_data])
    samples = samples + centre
    return samples, centre


def main() -> None:
    samples, centre = get_test_data(50)

    print(f"Expected centre: {centre :.5f}")

    print(f"Optimizer median: {MedianOptimizer()(samples) :.5f}")
    print(f"Optimizer mean: {MeanOptimizer()(samples) :.5f}")

    for p in [0.2, 0.5]:
        print(f"Optimizer SLSQP (p={p}): {SLSQPOptimizer(p)(samples)}")
        print(f"Optimizer Segment SLSQP (p={p}): {SegmentSLSQPOptimizer(p)(samples)}")

    for p in [3, 5]:
        print(f"Optimizer Bound (p={p}): {BoundOptimizer(p)(samples)}")


if __name__ == "__main__":
    main()
