from sklearn.datasets import load_wine

from lkmeans import LKMeans

data, targets = load_wine(as_frame=True, return_X_y=True)

lkmeans = LKMeans(n_clusters=2, p=2)
lkmeans.fit_predict(data)
print('Inertia', lkmeans.inertia_)
print('Centers', lkmeans.cluster_centers_)
