import numpy as np
import matplotlib.pyplot as plt

class KMeans:

    def __init__(self, X, k,visualize = False):
        self.data = X
        self.m, self.n = X.shape
        self.k = k

    # it would be nice to add cost calculation and selection of centroids based on that
    def run(self,iters):
        pass

    def run_k_means(self, iters):
        centroids = self.initiate_centroids()

        for i in range(0, iters):
            closest_centroids = self.closest_centroids(centroids)

            centroids = self.compute_centroids(closest_centroids)
        return centroids

    def closest_centroids(self, centroids):
        assigned_centroids = np.zeros((self.m, 1))
        for i in range(0, self.m):
            distances = np.zeros((self.k, 1))
            for j in range(0, self.k):
                distances[j] = np.sum(np.square((self.data[i, :] - centroids[j, :])))
            ix = np.argmin(distances)
            assigned_centroids[i] = ix
        return assigned_centroids

    def compute_centroids(self, closest_centroids):
        centroids=np.zeros((self.k,self.n))
        for i in range(0,self.k):
            temp = closest_centroids == i
            centroids[i, :] = (temp.T@self.data)/np.sum(temp)
        return centroids


    def initiate_centroids(self):
        rand_centr_idx = np.random.permutation(self.m)
        centroids = self.data[rand_centr_idx[:self.k, :], :]
        return centroids
