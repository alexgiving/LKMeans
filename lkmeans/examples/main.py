import time
from collections import defaultdict
from enum import Enum
from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score, completeness_score,
                             homogeneity_score, normalized_mutual_info_score, v_measure_score)
from tap import Tap

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


def main() -> None:
    args = ExperimentArguments(underscores_to_dashes=True).parse_args()

    _, prob, mu_list, cov_matrices = get_experiment_data(args.num_clusters, args.dimension)

    clustering = get_clustering_algorithm(args.clustering_algorithm)

    average_result = defaultdict(list)
    np.random.seed(5)

    for _ in range(args.repeats):

        clusters, labels, _ = generate_mix_distribution(
            probability=prob,
            mu_list=mu_list,
            cov_matrices=cov_matrices,
            n_samples=args.n_points,
            t=args.t_parameter
        )


        from sklearn.manifold import SpectralEmbedding, LocallyLinearEmbedding, Isomap, MDS
        import matplotlib.pyplot as plt


        X, _ = clusters, labels

        n_neighbors_list = [1, 5, 10, 99]
        num_methods = 6
        color_map = {3: 'red', 1: 'green', 0: 'blue', 3: 'yellow', 4: 'purple'}

        fig, axes = plt.subplots(len(n_neighbors_list), num_methods, figsize=(int(5.2*num_methods), 5*len(n_neighbors_list)))
        colors = [color_map[label] for label in labels]

        for n_neighbors, row_ax in zip(n_neighbors_list, axes):
            methods = {
                f"SE (NN={n_neighbors})": SpectralEmbedding(n_components=2, n_neighbors=n_neighbors),
                f"LLE (NN={n_neighbors})": LocallyLinearEmbedding(n_components=2, n_neighbors=n_neighbors),
                f"Isomap (NN={n_neighbors}, Minkowski p=1)": Isomap(n_components=2, n_neighbors=n_neighbors, p=1),
                f"Isomap (NN={n_neighbors}, Minkowski p=2)": Isomap(n_components=2, n_neighbors=n_neighbors, p=2),
                f"Isomap (NN={n_neighbors}, Minkowski p=5)": Isomap(n_components=2, n_neighbors=n_neighbors, p=5),
                f"MDS": MDS(n_components=2, random_state=42)
            }
            
            for ax, (name, method) in zip(row_ax, methods.items()):
                X_transformed = method.fit_transform(X)
                ax.scatter(X_transformed[:, 0], X_transformed[:, 1], c=colors)
                ax.set_title(name)

        fig.savefig(f"decompoition_chart_all.png", dpi=300, bbox_inches='tight')
        return

        if args.self_supervised_preprocessor_algorithm is not None:
            self_supervised_parameters = PreprocessorParameters(n_components=args.self_supervised_components)

            self_supervised_preprocessor = SelfSupervisedPreprocessor(
                args.self_supervised_preprocessor_algorithm,
                self_supervised_parameters
            )
            clusters = self_supervised_preprocessor.preprocess(clusters)

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
