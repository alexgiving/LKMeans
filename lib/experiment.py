import time
from pathlib import Path
from typing import List

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from lib.decomposition import get_tsne_clusters
from lib.experiment_metrics import get_average_experiment_metrics
from lib.kmeans import KMeans
from lib.metric_meter import MetricTable, insert_hline
from lib.points_generator import generate_mix_distribution
from lib.types import p_type


def get_covariance_matrix(sigma: float, dimension: int) -> np.ndarray:
    return np.eye(dimension) * sigma


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

    metrics = MetricTable()
    for t in distance_parameters:
        for p in minkowski_parameters:

            repeats_ari = []
            repeats_ami = []
            repeats_time = []

            for _ in range(repeats):

                clusters, labels = generate_mix_distribution(
                    probability=prob,
                    mu_list=mu_list,
                    cov_matrix_list=cov_matrix_list,
                    n_samples=n_points,
                    t=t
                )

                experiment_time = time.perf_counter()
                kmeans = KMeans(n_clusters=n_clusters, p=p)
                _, generated_labels = kmeans.fit(clusters)

                repeats_time.append(time.perf_counter()-experiment_time)
                repeats_ari.append(adjusted_rand_score(
                    labels, generated_labels))
                repeats_ami.append(adjusted_mutual_info_score(
                    labels, generated_labels))

            name = f'{experiment_name}, T:{t:.1f}, P:{p}'
            frame = get_average_experiment_metrics(
                repeats_ari, repeats_ami, name=name, time=repeats_time)
            metrics.add_frame(frame)

        if makes_plot:
            figure_name = f'factor_{t:.1f}'.replace('.', '_')
            fig = get_tsne_clusters(clusters, labels, None)
            fig.savefig(output_path / f'{figure_name}.png')
    print(metrics.get_table())

    table_name = 'experiment 1'
    table = metrics.get_latex_table(caption='Experiment 1')
    table = insert_hline(table)

    latex_logs = output_path / f'{table_name.replace(" ", "_")}.tex'
    with latex_logs.open('w') as f:
        f.write(table)
