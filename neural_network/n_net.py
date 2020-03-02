import numpy as np
from collections import deque


# this is my intepretation of neural network used for classification problem
# in this intepretation, in case of binary classification (yes or no ), you need to have two output nodes - one for yes and one for no
# I am aware that binary classification can be done with only one node but I wanted this class to be universal, you can use it with any number of output nodes you want
class NeuralNetwork:

    # n_layers - number of layers in the network (min number - 2)
    # n_units - list of number of units in each layers, length is equal to n_layers
    # e_init - epsilon used to randomly initialize parameters, should be small (eg. 0.1)
    def __init__(self, n_layers, n_units, e_init):
        if (n_layers != len(n_units)):
            raise Exception('n_layers and len(n_units) must be equal')
        if (e_init <= 0):
            raise Exception('e_init must be greater than 0')
        self.n_layers = n_layers
        self.n_units = n_units
        self.e_init = e_init
        self.__create_network()

    # creates a list of matrices (length=n_layers-1) that will be used to compute predictions
    # uses random initialization
    def __create_network(self):
        self.matrices = []
        for i in range(0, len(self.n_units) - 1):
            self.matrices.append(
                np.random.rand(self.n_units[i + 1], self.n_units[i] + 1) * 2 * self.e_init - self.e_init)

    def __cost(self, y_pred, y):
        if len(y) != len(y_pred):
            raise Exception('y and y_pred must have the same length')
        m = len(y)
        cost = (1 / m) * np.sum(np.sum(-y * np.log(y_pred) - (1 - y) * np.log(1 - y_pred)))
        return cost

    def __labels_to_vector(self, y):
        num_labels = self.n_units[self.n_layers - 1]
        m = len(y)
        y_vec = np.zeros((m, num_labels))
        for i in range(0, num_labels):
            y_vec[:, i][:, np.newaxis] = np.where(y == i, 1, 0)
        return y_vec

    # returns tuple (predicted labels, calculated probabilities)
    def predict(self, X):
        m = X.shape[0]
        units_vals = []
        a = X.copy()
        a = np.append(np.ones((m, 1)), a, 1)
        units_vals.append(a)
        for theta in self.matrices:
            z = a @ np.transpose(theta)
            a = self.__sigmoid(z)
            a = np.append(np.ones((m, 1)), a, 1)
            units_vals.append(a)
        units_vals[-1] = np.delete(units_vals[-1], 0, 1)

        return units_vals[len(units_vals) - 1].argmax(axis=1), units_vals

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # computing gradient using backpropagation algorithm
    def gradient_cost(self, X, y):
        # computing cost
        # computing gradient using backpropagation
        m = X.shape[0]
        y_vec = self.__labels_to_vector(y)
        pred = self.predict(X)[1]
        cost = self.__cost(pred[-1], y_vec)
        theta_grads = [np.zeros(t.shape) for t in self.matrices]
        y_v = self.__labels_to_vector(y)

        # compute errors
        sigmas = deque()
        sigmas.appendleft(pred[-1] - y_v)

        for i in reversed(range(1, len(pred) - 1)):
            sigmas.appendleft((sigmas[i - 1] @ self.matrices[i][:, 1:]) * pred[i][:, 1:] * (1 - pred[i][:, 1:]))

        # compute gradients
        for i, t in enumerate(theta_grads):
            t += np.transpose(sigmas[i]) @ pred[i]

        for i in range(0, len(theta_grads)):
            theta_grads[i] = theta_grads[i] / m

        return cost, theta_grads

    def gradient_descent(self, X, y, alpha, iters):
        costs = []
        for i in range(1, iters + 1):
            cost, theta_grads = self.gradient_cost(X, y)
            costs.append(cost)
            for i in range(0, len(self.matrices)):
                self.matrices[i] = self.matrices[i] - (alpha * theta_grads[i])
        return costs
