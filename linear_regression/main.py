import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression.lin_reg import LinearRegression
# first we will execute an univariate linear regression

#data1.txt is a data taken from Andrew Ng's Machine Learning course - Linear Regression
#in data1.txt, first column is the population of a city (in 10000s) and the second column is a profit of a food truck (int $10000s)


#showing the data
data=pd.read_csv('data1.txt',header=None)
data.columns=['Population','Profit']
print(data.head())

# visualising the data
data.columns=[0, 1] #changing columns' headers because plt.scatter doesnt see values propely if column's name is a string
plt.scatter(data[0],data[1])
plt.xlabel('Population of a City in 10000s')
plt.ylabel('Profit in $10000s')
plt.show()

