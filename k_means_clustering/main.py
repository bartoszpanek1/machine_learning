import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from k_means_clustering.k_means import  KMeans
data = loadmat("ex7data2.mat")
X= data["X"]

k_means = KMeans(X,3)
initial_centroids = np.array([[3,3],[6,2],[8,5]])
idx = k_means.closest_centroids(initial_centroids)
print(idx[0:3])

centroids = k_means.compute_centroids(idx)
print(centroids)




