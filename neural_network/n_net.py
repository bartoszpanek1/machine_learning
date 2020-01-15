import numpy as np


class NeuralNetwork:

    #n_layers - number of layers in the network (min number - 2)
    #n_units - list of number of units in each layers, length is equal to n_layers
    #e_init - epsilon used to randomly initialize parameters, should be small (eg. 0.1)
    def __init__(self,n_layers,n_units,e_init):
        self.n_layers=n_layers
        self.n_units=n_units
        self.e_init=e_init
        self.__create_network()


    #creates a list of matrices (length=n_layers-1) that will be used to compute predictions
    #uses random initialization
    def __create_network(self):
        self.matrices=[]
        for i in range(0,len(self.n_units)-1):
            self.matrices.append(np.random.rand(self.n_units[i+1],self.n_units[i]+1)*2*self.e_init-self.e_init)

    def predict(self,X):
        self.m=X.shape[0]
        self.units_vals=[]
        a = X
        self.units_vals.append(a)
        for theta in self.matrices:
            a=np.append(np.ones((self.m,1)),a,1)
            z=a@np.transpose(theta)
            a=self.__sigmoid(z)
            self.units_vals.append(a)
            print(a.shape)
        return self.units_vals[len(self.units_vals)-1].argmax(axis=1)




    def __sigmoid(self,z):
        return 1/(1+np.exp(-z))
