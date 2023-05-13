import time
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from lib.decomposition import get_tsne_clusters
from lib.lkmeans import LKMeans
from lib.metric_meter import (GraphicMeter, MetricMeter, MetricTable,
                              insert_hline)
from lib.points_generator import generate_mix_distribution


def get_covariance_matrix(sigma: float, dimension: int) -> np.ndarray:
    return np.eye(dimension) * sigma


# pylint: disable=too-many-arguments, too-many-locals
def repeat_iteration(
        repeats: int,
        n_clusters: int,
        n_points: int,
        prob: float,
        cov_matrices: list[np.ndarray],
        t: float,
        mu_list: list[np.ndarray],
        p: float | int,
        makes_plot: bool,
        output_path: Path

):
    repeat_metric_meter = MetricMeter()
    for _ in range(repeats):

        clusters, labels, centroids = generate_mix_distribution(
            probability=prob,
            mu_list=mu_list,
            cov_matrices=cov_matrices,
            n_samples=n_points,
            t=t
        )

        experiment_time = time.perf_counter()
        lkmeans = LKMeans(n_clusters=n_clusters, p=p)
        generated_labels = lkmeans.fit_predict(clusters)
        experiment_time = time.perf_counter() - experiment_time

        repeat_metric_meter.add_combination(
            ari=adjusted_rand_score(labels, generated_labels),
            ami=float(adjusted_mutual_info_score(labels, generated_labels)),
            inertia=lkmeans.inertia_,
            time=experiment_time
        )
        if makes_plot:
            figure_name = f'factor_{t:.1f}'.replace('.', '_')
            fig = get_tsne_clusters(clusters, labels, centroids)
            fig.savefig(
                str(output_path / f'{figure_name}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)
    average_ari, average_ami, average_inertia, average_time = repeat_metric_meter.get_average()
    return average_ari, average_ami, average_inertia, average_time


# pylint: disable= too-many-arguments, too-many-locals
def run_experiment(
        n_clusters: int,
        distance_parameters: list[float],
        minkowski_parameters: list[float | int],
        repeats: int,
        n_points: int,
        cov_matrices: list[np.ndarray],
        prob: float,
        mu_list: list[np.ndarray],
        experiment_name: str,
        output_path: Path,
        makes_plot: bool = False) -> None:
    '''Function for evaluation experiment'''

    output_path.mkdir(exist_ok=True, parents=True)

    table = MetricTable()

    generator = [GraphicMeter(distance_parameters, 't')
                 for _ in minkowski_parameters]
    graphic_t_metrics_dict = dict(zip(minkowski_parameters, generator))

    for t in distance_parameters:

        graphic_p_metrics = GraphicMeter(minkowski_parameters, 'p')
        for p in minkowski_parameters:

            average_ari, average_ami, average_inertia, average_time = repeat_iteration(
                repeats, n_clusters, n_points, prob,
                cov_matrices, t, mu_list, p, makes_plot, output_path
            )

            table.add_to_frame(
                ari=average_ari,
                ami=average_ami,
                inertia=average_inertia,
                time=average_time,
                name=f'{experiment_name}, T:{t:.1f}, P:{p}'
            )

            graphic_p_metrics.add_combination(
                ari=average_ari,
                ami=average_ami,
                inertia=average_inertia,
                time=average_time
            )

            graphic_t_metrics_dict[p].add_combination(
                ari=average_ari,
                ami=average_ami,
                inertia=average_inertia,
                time=average_time
            )

        for metric_graph in ['ARI', 'AMI', 'Inertia', 'Time']:
            figure_name = f'factor_{t:.1f}_{metric_graph}'.replace('.', '_')
            fig = graphic_p_metrics.get_graph(metric_graph)
            fig.savefig(
                str(output_path / f'{figure_name}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

    print(table.get_table())

    for p, graph_t_meter in graphic_t_metrics_dict.items():
        for metric in ['ARI', 'AMI', 'Inertia', 'Time']:
            figure_name = f'{metric}_by_t_with_p_{p}'.replace('.', '_')
            fig = graph_t_meter.get_graph(metric)
            fig.savefig(
                str(output_path / f'{figure_name}.png'), dpi=300, bbox_inches='tight')
            plt.close(fig)

    table_name = 'experiment 1'
    table = table.get_latex_table(caption='Experiment 1')
    table = insert_hline(table)

    latex_logs = output_path / f'{table_name.replace(" ", "_")}.tex'
    with latex_logs.open('w') as f:
        f.write(table)
