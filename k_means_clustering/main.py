import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from k_means_clustering.k_means import  KMeans

# example data is taken from the Andrew Ng's machine learning course

data = loadmat("data.mat")
X= data["X"]

k_means = KMeans(X,3,visualize=True)
centroids_positions = k_means.run_k_means(5)
print(centroids_positions)







