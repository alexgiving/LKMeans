import time
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from lib.decomposition import get_tsne_clusters
from lib.experiment_metrics import get_average_experiment_metrics
from lib.kmeans import KMeans
from lib.metric_meter import (GraphicMeter, MetricMeter, MetricTable,
                              insert_hline)
from lib.points_generator import generate_mix_distribution
from lib.types import p_type


def get_covariance_matrix(sigma: float, dimension: int) -> np.ndarray:
    return np.eye(dimension) * sigma


# pylint: disable= too-many-arguments, too-many-locals
def run_experiment(
        dimension: int,
        n_clusters: int,
        distance_parameters: List[float],
        minkowski_parameters: List[p_type],
        repeats: int,
        n_points: int,
        sigma_list: List[p_type],
        prob: float,
        mu_list: List[np.ndarray],
        experiment_name: str,
        output_path: Path,
        makes_plot: bool = False) -> None:
    '''Function for evaluation experiment'''

    output_path.mkdir(exist_ok=True, parents=True)

    cov_matrix_list = [get_covariance_matrix(
        sigma, dimension) for sigma in sigma_list]

    table = MetricTable()
    for t in distance_parameters:

        graphic_metrics = GraphicMeter(minkowski_parameters)
        for p in minkowski_parameters:

            repeat_metric_meter = MetricMeter()
            for _ in range(repeats):

                clusters, labels, centroids = generate_mix_distribution(
                    probability=prob,
                    mu_list=mu_list,
                    cov_matrix_list=cov_matrix_list,
                    n_samples=n_points,
                    t=t
                )

                experiment_time = time.perf_counter()
                kmeans = KMeans(n_clusters=n_clusters, p=p)
                centroids, generated_labels = kmeans.fit(clusters)
                experiment_time = time.perf_counter() - experiment_time

                repeat_metric_meter.add_ami(float(adjusted_mutual_info_score(
                    labels, generated_labels)))
                repeat_metric_meter.add_ari(adjusted_rand_score(
                    labels, generated_labels))
                repeat_metric_meter.add_inertia(
                    kmeans.inertia(clusters, centroids))
                repeat_metric_meter.add_time(experiment_time)

            average_ari, average_ami, average_inertia, average_time = repeat_metric_meter.get_average()

            name = f'{experiment_name}, T:{t:.1f}, P:{p}'
            frame = get_average_experiment_metrics(
                average_ari, average_ami, average_inertia, name=name, time=average_time)
            table.add_frame(frame)

            graphic_metrics.add_ari(average_ari)
            graphic_metrics.add_ami(average_ami)
            graphic_metrics.add_inertia(average_inertia)
            graphic_metrics.add_time(average_time)

        for metric_graph in ['ARI', 'AMI', 'Inertia', 'Time']:
            figure_name = f'factor_{t:.1f}_{metric_graph}'.replace('.', '_')
            fig = graphic_metrics.get_graph(metric_graph)
            fig.savefig(str(output_path / f'{figure_name}.png'))
        # if makes_plot:
        #     figure_name = f'factor_{t:.1f}'.replace('.', '_')
        #     fig = get_tsne_clusters(clusters, labels, centroids)
        #     fig.savefig(output_path / f'{figure_name}.png')

    print(table.get_table())

    table_name = 'experiment 1'
    table = table.get_latex_table(caption='Experiment 1')
    table = insert_hline(table)

    latex_logs = output_path / f'{table_name.replace(" ", "_")}.tex'
    with latex_logs.open('w') as f:
        f.write(table)
