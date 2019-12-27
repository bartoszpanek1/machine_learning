import numpy as np
class LinearRegression:

    def __init__(self,data_X,data_y,lambda_reg=0): #regularization parameter lambda is optional (someone may not want to regularize linear regression)
        self.X=data_X
        self.y=data_y
        self.lambda_reg=lambda_reg
        self.theta=np.zeros((self.X.shape[1]+1,1))
        self.m=self.X.shape[0] # number of training examples

    #using a Mean Squared Error cost function
    def compute_cost(self):
        cost = np.square((self.X*self.theta - self.y))
        if self.lambda_reg>0: # if regularization parameter lambda is greater than 0 we need to apply regularization
            reg_term = self.lambda_reg * (np.transpose(self.theta) @ self.theta)
            cost = cost + reg_term
        return cost


