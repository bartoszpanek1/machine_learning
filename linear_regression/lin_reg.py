import numpy as np


class LinearRegression:

    def __init__(self, data_X, data_y,
                 lambda_reg=0):  # regularization parameter lambda is optional (someone may not want to regularize linear regression)
        self.m = data_X.shape[0]  # number of training examples
        self.X = np.append(np.ones((self.m, 1)), data_X, 1)  # training data
        self.y = data_y  # training answers to the given data
        self.lambda_reg = lambda_reg  # regularization parameter lambda
        self.theta = np.zeros((self.X.shape[1], 1))  # initial parameters - all zeros
        self.cost = self.__compute_cost()  # initial cost based on initial theta

    # using a Mean Squared Error cost function
    def __compute_cost(self):
        cost = np.sum(np.square((self.X @ self.theta - self.y))) / (2 * self.m)
        if self.lambda_reg > 0:  # if regularization parameter lambda is greater than 0 we need to apply regularization
            reg_term = self.lambda_reg * (np.transpose(self.theta) @ self.theta)
            cost = cost + reg_term
        return cost

    # using Gradient Descent algorithm to get optimal parameters theta
    # function returns list of costs (length the same as gradient descent's number of iterations)
    # iters - number of iterations for GD, needs to be big enough to see a convergence, not useful to be very big because GD will return similar values anyway and will work longer
    # alpha - learning rate, it can't be too small because GD will be slow to converge and it ca't be too big because GD would fail to converge
    def compute_optimal_theta(self, iters, alpha):
        costs = []
        for i in range(1, iters + 1):
            self.theta = self.theta * (1 - alpha * self.lambda_reg / self.m) - (alpha / self.m) * (
                        np.transpose(self.X) @ (self.X @ self.theta - self.y))
            costs.append(self.__compute_cost())
        return costs
