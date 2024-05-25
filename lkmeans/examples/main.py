import time
from collections import defaultdict
from enum import Enum
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score, adjusted_rand_score
from tap import Tap

from lkmeans.clustering import HardSSLKMeans, LKMeans, SoftSSLKMeans
from lkmeans.clustering.base import Clustering
from lkmeans.clustering.supervised.utils import select_supervisor_targets
from lkmeans.examples.data.experiment_data import get_experiment_data
from lkmeans.examples.data.points_generator import generate_mix_distribution


class ClusteringAlgorithmType(Enum):
    LKMEANS = 'lkmeans'
    SOFT_SS_LKMEANS = 'soft_ss_lkmeans'
    HARD_SS_LKMEANS = 'hard_ss_lkmeans'


class ExperimentArguments(Tap):
    minkowski_parameter: float
    t_parameter: float
    n_points: int
    clustering_algorithm: ClusteringAlgorithmType = ClusteringAlgorithmType.LKMEANS

    num_clusters: int = 2
    dimension: int = 20
    repeats: int = 10
    supervision_ratio: float = 0


def get_clustering_algorithm(clustering_type: ClusteringAlgorithmType) -> Clustering:
    clustering_map: Dict[clustering_type, Clustering] = {
        ClusteringAlgorithmType.LKMEANS: LKMeans,
        ClusteringAlgorithmType.SOFT_SS_LKMEANS: SoftSSLKMeans,
        ClusteringAlgorithmType.HARD_SS_LKMEANS: HardSSLKMeans
    }
    return clustering_map[clustering_type]


def calculate_metrics(labels: NDArray, generated_labels: NDArray) -> Dict[str, float]:
    return {
        'ari': float(adjusted_rand_score(labels, generated_labels)),
        'ami': float(adjusted_mutual_info_score(labels, generated_labels)),
        'accuracy': float(accuracy_score(labels, generated_labels)),
    }


def main() -> None:
    args = ExperimentArguments(underscores_to_dashes=True).parse_args()

    _, prob, mu_list, cov_matrices = get_experiment_data(args.num_clusters, args.dimension)

    clustering = get_clustering_algorithm(args.clustering_algorithm)

    average_result = defaultdict(list)

    for _ in range(args.repeats):

        clusters, labels, _ = generate_mix_distribution(
            probability=prob,
            mu_list=mu_list,
            cov_matrices=cov_matrices,
            n_samples=args.n_points,
            t=args.t_parameter
        )

        lkmeans = clustering(n_clusters=args.num_clusters, p=args.minkowski_parameter)

        if args.clustering_algorithm is ClusteringAlgorithmType.LKMEANS:

            experiment_time = time.perf_counter()
            generated_labels = lkmeans.fit_predict(clusters)
        else:
            experiment_time = time.perf_counter()
            supervisor_targets = select_supervisor_targets(labels, args.supervision_ratio)
            generated_labels = lkmeans.fit_predict(clusters, supervisor_targets)
        experiment_time = time.perf_counter() - experiment_time

        metrics_dict = calculate_metrics(
            labels=labels,
            generated_labels=generated_labels,
        )
        result = {**metrics_dict, 'time': experiment_time, 'inertia': lkmeans.inertia_}
        for key, value in result.items():
            average_result[key].append(value)
    for key, value in result.items():
        average_result[key] = np.mean(value)
    print(dict(average_result))


if __name__ == '__main__':
    main()
