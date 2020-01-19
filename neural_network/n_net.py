import numpy as np

#this is my intepretation of neural network used for classification problem
# in this intepretation, in case of binary classification (yes or no ), you need to have two output nodes - one for yes and one for no
# I am aware that binary classification can be done with only one node but I wanted this class to be universal, you can use it with any number of output nodes you want
class NeuralNetwork:

    # n_layers - number of layers in the network (min number - 2)
    # n_units - list of number of units in each layers, length is equal to n_layers
    # e_init - epsilon used to randomly initialize parameters, should be small (eg. 0.1)
    def __init__(self, n_layers, n_units, e_init):
        if(n_layers!=len(n_units)):
            raise Exception('n_layers and len(n_units) must be equal')
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


    def cost(self, y_pred,y):
        if(len(y)!=len(y_pred)):
            raise Exception('y and y_pred must have the same length')
        num_labels=self.n_units[self.n_layers-1]
        m=len(y)
        y_vec=np.zeros((m,num_labels))
        for i in range(0,num_labels):
            y_vec[:,i][:,np.newaxis]=np.where(y==i,1,0)
        cost=(1/m)*np.sum(np.sum(-y_vec*np.log(y_pred) - (1-y_vec)*np.log(1-y_pred)))
        return cost

#returns tuple (predicted labels, calculated probabilities)
    def predict(self, X):
        m = X.shape[0]
        units_vals = []
        a = X
        units_vals.append(a)
        for theta in self.matrices:
            a = np.append(np.ones((m, 1)), a, 1)
            z = a @ np.transpose(theta)
            a = self.__sigmoid(z)
            units_vals.append(a)
        return units_vals[len(units_vals) - 1].argmax(axis=1),units_vals[len(units_vals) - 1]

    def __sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
