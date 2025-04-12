import time
from collections import defaultdict
from enum import Enum
from typing import Dict, Optional, Type

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    adjusted_mutual_info_score,
    adjusted_rand_score,
    completeness_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)
from tap import Tap

from lkmeans.clustering import (
    HardSemiSupervisedLKMeans,
    LKMeans,
    SoftSemiSupervisedLKMeans,
)
from lkmeans.clustering.base import Clustering
from lkmeans.clustering.self_supervised.preprocessor import SelfSupervisedPreprocessor
from lkmeans.clustering.self_supervised.preprocessor_parameters import (
    PreprocessorParameters,
)
from lkmeans.clustering.self_supervised.preprocessor_type import PreprocessorType
from lkmeans.clustering.semi_supervised.utils import select_supervisor_targets
from lkmeans.examples.data.experiment_data import get_experiment_data
from lkmeans.examples.data.points_generator import generate_mix_distribution


class ClusteringAlgorithmType(Enum):
    LKMEANS = "lkmeans"
    SOFT_SEMI_SUPERVISED_LKMEANS = "soft_semi_supervised_lkmeans"
    HARD_SEMI_SUPERVISED_LKMEANS = "hard_semi_supervised_lkmeans"


class ExperimentArguments(Tap):
    minkowski_parameter: float
    t_parameter: float
    n_points: int
    clustering_algorithm: ClusteringAlgorithmType = ClusteringAlgorithmType.LKMEANS
    self_supervised_preprocessor_algorithm: Optional[PreprocessorType] = None
    self_supervised_components: int = 2

    num_clusters: int = 2
    dimension: int = 20
    repeats: int = 10
    supervision_ratio: float = 0


def get_clustering_algorithm(clustering_type: ClusteringAlgorithmType) -> Type[Clustering]:
    clustering_map: Dict[ClusteringAlgorithmType, Type[Clustering]] = {
        ClusteringAlgorithmType.LKMEANS: LKMeans,
        ClusteringAlgorithmType.SOFT_SEMI_SUPERVISED_LKMEANS: SoftSemiSupervisedLKMeans,
        ClusteringAlgorithmType.HARD_SEMI_SUPERVISED_LKMEANS: HardSemiSupervisedLKMeans,
    }
    return clustering_map[clustering_type]


def calculate_metrics(labels: NDArray, generated_labels: NDArray) -> Dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(labels, generated_labels)),
        "ami": float(adjusted_mutual_info_score(labels, generated_labels)),
        "completeness": float(completeness_score(labels, generated_labels)),
        "homogeneity": float(homogeneity_score(labels, generated_labels)),
        "nmi": float(normalized_mutual_info_score(labels, generated_labels)),
        "v_measure": float(v_measure_score(labels, generated_labels)),
        "accuracy": float(accuracy_score(labels, generated_labels)),
    }


def main() -> None:
    args = ExperimentArguments(underscores_to_dashes=True).parse_args()

    _, prob, mu_list, cov_matrices = get_experiment_data(args.num_clusters, args.dimension)

    clustering = get_clustering_algorithm(args.clustering_algorithm)

    experiment_results = defaultdict(list)
    average_result: Dict[str, float] = {}

    for _ in range(args.repeats):

        clusters, labels, _ = generate_mix_distribution(
            probability=prob,
            mu_list=mu_list,
            cov_matrices=cov_matrices,
            n_samples=args.n_points,
            t=args.t_parameter,
        )

        if args.self_supervised_preprocessor_algorithm is not None:
            self_supervised_parameters = PreprocessorParameters(
                n_components=args.self_supervised_components
            )

            self_supervised_preprocessor = SelfSupervisedPreprocessor(
                args.self_supervised_preprocessor_algorithm, self_supervised_parameters
            )
            clusters = self_supervised_preprocessor.preprocess(clusters)

        lkmeans = clustering(n_clusters=args.num_clusters, p=args.minkowski_parameter)

        experiment_time = time.perf_counter()
        if args.clustering_algorithm is ClusteringAlgorithmType.LKMEANS:
            generated_labels = lkmeans.fit_predict(clusters)
        else:
            supervisor_targets = select_supervisor_targets(labels, args.supervision_ratio)
            generated_labels = lkmeans.fit_predict(clusters, supervisor_targets)
        experiment_time = time.perf_counter() - experiment_time

        metrics_dict = calculate_metrics(
            labels=labels,
            generated_labels=generated_labels,
        )
        result = {**metrics_dict, "time": experiment_time, "inertia": lkmeans.inertia_}
        for key, value in result.items():
            experiment_results[key].append(value)
    for key, value in experiment_results.items():
        average_result[key] = np.mean(value)
    print(dict(average_result))


if __name__ == "__main__":
    main()
