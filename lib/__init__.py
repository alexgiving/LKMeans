from lib.decomposition import get_tsne_clusters
from lib.experiment_metrics import get_average_experiment_metrics
from lib.kmeans import KMeans
from lib.metric_meter import MetricTable, insert_hline
from lib.points_generator import (generate_cluster_centroids,
                                  generate_gaussian_clusters)
from lib.experiment import run_experiment