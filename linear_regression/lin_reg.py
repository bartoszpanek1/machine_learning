import numpy as np


class LinearRegression:

    def __init__(self, data_X, data_y,
                 # regularization parameter lambda is optional (someone may not want to regularize linear regression if data is simple)
                 lambda_reg=0,
                 # normalization is optional ( but necessary in big data sets - prevents overflow and speeds up computation )
                 normalize=False):
        self.m = data_X.shape[0]  # number of training examples
        self.y = data_y  # training answers to the given data
        self.normalized = normalize
        self.X_mean = None
        self.X_std_var = None
        if not normalize:
            self.X = np.append(np.ones((self.m, 1)), data_X, 1)  # training data not normalized
        else:
            self.X = np.append(np.ones((self.m, 1)), self.normalize_features(data_X), 1)  # training data normalized
        self.lambda_reg = lambda_reg  # regularization parameter lambda
        self.theta = np.zeros((self.X.shape[1], 1))  # initial parameters - all zeros
        self.cost = self.__compute_cost()  # initial cost based on initial theta

    # using a Mean Squared Error cost function
    def __compute_cost(self):
        cost = np.sum(np.square((self.X @ self.theta - self.y))) / (2 * self.m)
        if self.lambda_reg > 0:  # if regularization parameter lambda is greater than 0 we need to apply regularization
            reg_term = self.lambda_reg * ((np.transpose(self.theta) @ self.theta)[0][0])
            cost = cost + reg_term
        return cost

    # using Gradient Descent algorithm to get optimal parameters theta
    # function returns list of costs (length the same as gradient descent's number of iterations)
    # iters - number of iterations for GD, needs to be big enough to see a convergence, not useful to be very big because GD will return similar values anyway and will work longer
    # alpha - learning rate, it can't be too small because GD will be slow to converge and it ca't be too big because GD would fail to converge
    def compute_optimal_theta(self, iters, alpha):
        costs = []
        for i in range(1, iters + 1):
            if self.lambda_reg==0:
                self.theta = self.theta - (alpha / self.m) * (
                        np.transpose(self.X) @ (self.X @ self.theta - self.y))
            costs.append(self.__compute_cost())
        self.cost = costs[-1]
        return costs

    def normalize_features(self, X):
        mean = np.mean(X, axis=0)
        std_var = np.std(X, axis=0)
        X_normalized = (X - mean) / std_var
        self.X_mean = mean
        self.X_std_var = std_var
        return X_normalized

    def predict(self, data):
        def normalize(d):
            return (d - self.X_mean) / self.X_std_var

        return ((np.append(np.ones((1, 1)), normalize(data), 1)) @ self.theta)[0][
            0] if self.normalized else ((np.append(np.ones((1, 1)), data, 1)) @ self.theta)[0][0]


#added simple function to generate data and write it to the file It is not ideal, need to scale parameters properly to generate good data
def simple_generate_data(pol_degree, y_parameters, no_of_examples):
    np.random.seed(0)
    x = 5 - 4 * np.random.normal(0, 1, no_of_examples)
    y = np.random.normal(-90, 90, no_of_examples)
    for i in range(0, pol_degree):
        y = y + y_parameters[i] * (x ** (i + 1))
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    np.savetxt('generated_data.txt',np.append(x,y,axis=1),delimiter=',')



