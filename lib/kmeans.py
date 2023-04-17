import numpy as np
from scipy.optimize import minimize, root

from lib.minkowski import (minkowski_1_derivative, minkowski_distance,
                           minkowski_loss)


class KMeans:
    def __init__(self, n_clusters, max_iter=100, p=2):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.p = p
        self.centroids = []
        self.labels = []

    def fit(self, X):
        # initialize centroids randomly
        self.centroids = [X[i] for i in np.random.choice(X.shape[0], self.n_clusters, replace=False)]

        for _ in range(self.max_iter):
            clusters = [[] for _ in range(self.n_clusters)]
            self.labels = []

            # assign each data point to the closest centroid
            for x in X:
                distances_to_each_cebtroid = [minkowski_distance(x, centroid, self.p) for centroid in self.centroids]
                closest_centroid = np.argmin(distances_to_each_cebtroid)
                clusters[closest_centroid].append(x)
                self.labels.append(closest_centroid)

            # update centroids using the specified optimizer
            for cluster_id, cluster in enumerate(clusters):
                cluster = np.array(cluster)
                if len(cluster) == 0:
                    continue
                if self.p == 2:
                    self.centroids[cluster_id] = np.mean(cluster, axis=0)
                elif self.p == 1:
                    self.centroids[cluster_id] = np.median(cluster, axis=0)
                elif self.p > 1:
                    '''
                    SGD optimizer
                    Amorim, Renato. (2012). Feature Weighting for Clustering: Using K-Means and the Minkowski Metric. 
                    '''
                    learning_rate = 0.01
                    grad_descent = 0.1
                    n_iters = 50
                    centroid = np.mean(cluster, axis=0)

                    for sgd_iteration in range(n_iters):
                        if sgd_iteration == n_iters / 2:
                            learning_rate *= grad_descent
                        elif sgd_iteration == n_iters / 4:
                            learning_rate *= grad_descent
                        loss = minkowski_loss(cluster, centroid, self.p)
                        grad = np.gradient(loss, axis=0)
                        centroid -= learning_rate * grad
                    self.centroids[cluster_id] = centroid
                elif 0 < self.p < 1:
                    '''
                    Find extremum of minkowski function by root of 1 derivative.
                    '''
                    sol = root(minkowski_1_derivative, np.mean(cluster, axis=0), args=(cluster, self.p), method='hybr')
                    self.centroids[cluster_id] = sol.x
                else:
                    raise ValueError(f'Unsupported value of p: {self.p}')
                    # if self.optimizer == 'SLSQP':
                    #     bounds = [(None, None)] * self.centroids[j].shape[0]
                    #     self.centroids[j] = minimize(
                    #         lambda x: minkowski_distance(x, cluster, self.p),
                    #         self.centroids[j].flatten(),
                    #         method=self.optimizer,
                    #         bounds=bounds
                    #     ).x.copy()
                    # elif self.optimizer in ('Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'trust-constr'):
                    #     self.centroids[j] = minimize(lambda x: minkowski_distance(x, cluster, self.p), self.centroids[j], method=self.optimizer).x.copy()

        return self.centroids, self.labels