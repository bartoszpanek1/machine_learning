import numpy as np
import matplotlib.pyplot as plt


class KMeans:
    # m - number of training examples
    # n - number of dimensions
    # k - number of clusters
    def __init__(self, X, k, visualize=False):
        self.data = X
        self.m, self.n = X.shape
        self.k = k
        self.visualize = True
        if self.visualize and self.n != 2:
            raise Exception("Visualization is only possible with 2nd dimension (n = 2) ")

    def run_k_means(self, iters):
        centroids = self.initiate_centroids()
        if self.visualize:
            self.visualize_data(centroids, self.closest_centroids(centroids), 'Initial positions')
        for i in range(0, iters):
            closest_centroids = self.closest_centroids(centroids)

            centroids = self.compute_centroids(closest_centroids)
            if self.visualize:
                self.visualize_data(centroids, closest_centroids, 'Iteration ' + str(i + 1))
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

    # computing the mean coordinates of assigned points and updating the position of a centroid using that data
    def compute_centroids(self, closest_centroids):
        centroids = np.zeros((self.k, self.n))
        for i in range(0, self.k):
            temp = closest_centroids == i
            centroids[i, :] = (temp.T @ self.data) / np.sum(temp)
        return centroids

    # Centroids have random positions at the beginning
    def initiate_centroids(self):
        rand_centr_idx = np.random.permutation(self.m)
        print(rand_centr_idx.shape)
        centroids = self.data[rand_centr_idx[:self.k], :]
        return centroids

    def visualize_data(self, centroids, closest_centroids, title):
        color = "rgb"
        for k in range(0, self.k):
            grp = (closest_centroids == k).reshape(self.m, 1)
            plt.scatter(self.data[grp[:, 0], 0], self.data[grp[:, 0], 1], c=color[k], s=15)
        plt.scatter(centroids[:, 0], centroids[:, 1], s=120, marker="D", c="black", linewidth=5)
        plt.title(title)
        plt.show()
