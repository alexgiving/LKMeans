import time
from pathlib import Path
from typing import List, Union

from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

from lib.decomposition import get_tsne_clusters
from lib.experiment_metrics import get_average_experiment_metrics
from lib.kmeans import KMeans
from lib.metric_meter import MetricTable, insert_hline
from lib.points_generator import (generate_cluster_centroids,
                                  generate_gaussian_clusters)


def run_experiment(
        dimension: int,
        n_clusters: int,
        distance_parameters: List[float],
        minkowski_parameters: List[Union[int, float]],
        repeats: int,
        n_points: int,
        experiment_name: str,
        output_path: Path) -> None:
    '''Function for evaluation experiment'''

    output_path.mkdir(exist_ok=True, parents=True)

    metrics = MetricTable()
    for distance_factor in distance_parameters:
        distance_factor_name = f'{distance_factor:.1f}'

        for p in minkowski_parameters:

            repeats_ari = []
            repeats_ami = []
            repeats_time = []

            for _ in range(repeats):

                centroid = generate_cluster_centroids(
                    dimension=dimension,
                    n_clusters=n_clusters,
                    distance_factor=distance_factor
                )

                clusters, labels = generate_gaussian_clusters(
                    dimension=dimension,
                    sigma=1,
                    centroid_locations=centroid,
                    n_points_per_cluster=n_points
                )

                experiment_time = time.perf_counter()
                kmeans = KMeans(n_clusters=n_clusters, p=p)
                _, generated_labels = kmeans.fit(clusters)

                repeats_time.append(time.perf_counter()-experiment_time)
                repeats_ari.append(adjusted_rand_score(labels, generated_labels))
                repeats_ami.append(adjusted_mutual_info_score(labels, generated_labels))

            frame = get_average_experiment_metrics(repeats_ari, repeats_ami, name=experiment_name, time=repeats_time)
            metrics.add_frame(frame)

        # To add midrule
        metrics.add_empty_frame(True)

        log_name = f'factor_{distance_factor_name}'.replace('.', '_')
        fig = get_tsne_clusters(clusters, labels, centroid)
        fig.savefig(output_path / f'{log_name}.png')

    table_name = 'experiment 1'
    table = metrics.get_latex_table(caption='Experiment 1')
    table = insert_hline(table)

    latex_logs = output_path / f'{table_name.replace(" ", "_")}.tex'
    with latex_logs.open('w') as f:
        f.write(table)
