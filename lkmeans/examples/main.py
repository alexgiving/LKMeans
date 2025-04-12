import time
from collections import defaultdict
from enum import Enum
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
                             homogeneity_score, normalized_mutual_info_score, v_measure_score)
from tap import Tap
from sklearn.datasets import fetch_openml
from lkmeans.clustering import HardSemiSupervisedLKMeans, LKMeans, SoftSemiSupervisedLKMeans
from lkmeans.clustering.base import Clustering
from lkmeans.clustering.self_supervised.preprocessor import SelfSupervisedPreprocessor
from lkmeans.clustering.self_supervised.preprocessor_parameters import PreprocessorParameters
from lkmeans.clustering.self_supervised.preprocessor_type import PreprocessorType
from lkmeans.clustering.semi_supervised.utils import select_supervisor_targets
from lkmeans.examples.data.experiment_data import get_experiment_data
from lkmeans.examples.data.points_generator import generate_mix_distribution


class ClusteringAlgorithmType(Enum):
    LKMEANS = 'lkmeans'
    SOFT_SEMI_SUPERVISED_LKMEANS = 'soft_semi_supervised_lkmeans'
    HARD_SEMI_SUPERVISED_LKMEANS = 'hard_semi_supervised_lkmeans'


class DataType(Enum):
    GENERATED = 'generated'
    WINE = 'wine'
    BREAST_CANCER = 'breast_cancer'
    IRIS = 'iris'
    DIGITS = 'digits'
    MNIST = 'mnist'
    CIFAR10 = "cifar10"


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

    data_type: DataType = DataType.GENERATED


def get_clustering_algorithm(clustering_type: ClusteringAlgorithmType) -> Clustering:
    clustering_map: Dict[ClusteringAlgorithmType, Clustering] = {
        ClusteringAlgorithmType.LKMEANS: LKMeans,
        ClusteringAlgorithmType.SOFT_SEMI_SUPERVISED_LKMEANS: SoftSemiSupervisedLKMeans,
        ClusteringAlgorithmType.HARD_SEMI_SUPERVISED_LKMEANS: HardSemiSupervisedLKMeans
    }
    return clustering_map[clustering_type]


def calculate_metrics(labels: NDArray, generated_labels: NDArray) -> Dict[str, float]:
    return {
        'ari': float(adjusted_rand_score(labels, generated_labels)),
        'ami': float(adjusted_mutual_info_score(labels, generated_labels)),
        'completeness': float(completeness_score(labels, generated_labels)),
        'homogeneity': float(homogeneity_score(labels, generated_labels)),
        'nmi': float(normalized_mutual_info_score(labels, generated_labels)),
        'v_measure': float(v_measure_score(labels, generated_labels)),
        'accuracy': float(accuracy_score(labels, generated_labels)),
    }


def generate_data(args) -> ExperimentArguments:
    if args.data_type is DataType.GENERATED:
        _, prob, mu_list, cov_matrices = get_experiment_data(args.num_clusters, args.dimension)
        
        data, labels, _ = generate_mix_distribution(
            probability=prob,
            mu_list=mu_list,
            cov_matrices=cov_matrices,
            n_samples=args.n_points,
            t=args.t_parameter
        )
    elif args.data_type is DataType.WINE:
        data, labels = datasets.load_breast_cancer(return_X_y=True)
    elif args.data_type is DataType.BREAST_CANCER:
        data, labels = datasets.load_breast_cancer(return_X_y=True)
    elif args.data_type is DataType.IRIS:
        data, labels = datasets.load_iris(return_X_y=True)
    elif args.data_type is DataType.DIGITS:
        data, labels = datasets.load_digits(return_X_y=True)

    elif args.data_type is DataType.MNIST:
        data, labels = fetch_openml('mnist_784', version=1, return_X_y=True)
        labels = labels.astype(int)
    elif args.data_type is DataType.CIFAR10:
        data, labels = fetch_openml('CIFAR_10_small', version=1, return_X_y=True)
        labels = labels.astype(int)

    num_clusters_in_dataset = len(set(labels))
    if args.num_clusters != num_clusters_in_dataset:
        print(f"Warning: {args.data_type} has {num_clusters_in_dataset} clusters while num_clusters = {args.num_clusters} is passed. We change the num_clusters to {num_clusters_in_dataset}")
        args.num_clusters = num_clusters_in_dataset
    return data, labels


def main() -> None:
    args = ExperimentArguments(underscores_to_dashes=True).parse_args()

    clustering = get_clustering_algorithm(args.clustering_algorithm)

    average_result = defaultdict(list)

    for _ in range(args.repeats):

        clusters, labels = generate_data(args)

        if args.self_supervised_preprocessor_algorithm is not None:
            self_supervised_parameters = PreprocessorParameters(n_components=args.self_supervised_components)

            self_supervised_preprocessor = SelfSupervisedPreprocessor(
                args.self_supervised_preprocessor_algorithm,
                self_supervised_parameters
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
        result = {**metrics_dict, 'time': experiment_time, 'inertia': lkmeans.inertia_}
        for key, value in result.items():
            average_result[key].append(value)
    for key, value in average_result.items():
        average_result[key] = np.mean(value)
    print(dict(average_result))


if __name__ == '__main__':
    main()
