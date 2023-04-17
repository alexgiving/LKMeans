import time
from pathlib import Path
from typing import List, Union

import numpy as np
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from lib.decomposition import get_tsne_clusters
from lib.experiment_metrics import get_average_experiment_metrics
from lib.kmeans import KMeans
from lib.metric_meter import MetricTable, insert_hline
from lib.points_generator import generate_2_mix_distribution


def run_experiment(
        dimension: int,
        n_clusters: int,
        distance_parameters: List[float],
        minkowski_parameters: List[Union[int, float]],
        repeats: int,
        n_points: int,
        experiment_name: str,
        output_path: Path,
        makes_plot: bool = False) -> None:
    '''Function for evaluation experiment'''

    output_path.mkdir(exist_ok=True, parents=True)

    metrics = MetricTable()
    for t in distance_parameters:
        for p in minkowski_parameters:

            repeats_ari = []
            repeats_ami = []
            repeats_time = []

            for _ in range(repeats):

                sigma_1 = 1
                sigma_2 = 1

                clusters, labels = generate_2_mix_distribution(
                    probability=0.5,
                    mu_1=np.array([[-2, 0] + [0] * (dimension-2)]),
                    mu_2=np.array([[2, 0] + [0] * (dimension-2)]),
                    cov_matrix_1=np.eye(dimension) * sigma_1,
                    cov_matrix_2=np.eye(dimension) * sigma_2,
                    n_samples=n_points,
                    t=t
                    )

                experiment_time = time.perf_counter()
                kmeans = KMeans(n_clusters=n_clusters, p=p)
                _, generated_labels = kmeans.fit(clusters)

                repeats_time.append(time.perf_counter()-experiment_time)
                repeats_ari.append(adjusted_rand_score(labels, generated_labels))
                repeats_ami.append(adjusted_mutual_info_score(labels, generated_labels))

            name = f'{experiment_name}, T:{t:.1f}, P:{p}'
            frame = get_average_experiment_metrics(repeats_ari, repeats_ami, name=name, time=repeats_time)
            metrics.add_frame(frame)

        # To add midrule
        metrics.add_empty_frame(True)

        if makes_plot:
            figure_name = f'factor_{t:.1f}'.replace('.', '_')
            fig = get_tsne_clusters(clusters, labels, None)
            fig.savefig(output_path / f'{figure_name}.png')

    table_name = 'experiment 1'
    table = metrics.get_latex_table(caption='Experiment 1')
    table = insert_hline(table)

    latex_logs = output_path / f'{table_name.replace(" ", "_")}.tex'
    with latex_logs.open('w') as f:
        f.write(table)
