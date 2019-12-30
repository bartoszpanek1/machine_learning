import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression.lin_reg import LinearRegression

# first we will execute an univariate linear regression

# data1.txt is a data taken from Andrew Ng's Machine Learning course - Linear Regression
# in data1.txt, first column is the population of a city (in 10000s) and the second column is a profit of a food truck (int $10000s)

# -----UNIVARIATE DATA-----
# showing the data
print('----------------------')
print('------UNIVARIATE------')
print('----------------------')
data = pd.read_csv('data1.txt', header=None)
data.columns = ['Population', 'Profit']

print('First 5 examples from data1.txt')
print(data.head())

# visualising the data
data.columns = [0,
                1]  # changing columns' headers because plt.scatter doesnt see values propely if column's name is a string
plt.scatter(data[0], data[1])
plt.xlabel('Population of a City in 10000s')
plt.ylabel('Profit in $10000s')
plt.title('Training data')
plt.show()

input('PRESS ENTER TO APPLY LINEAR REGRESSION \n')
X = data[0].copy(deep=True).to_numpy().reshape((data[0].shape[0], 1))
y = data[1].copy(deep=True).to_numpy().reshape((data[1].shape[0], 1))

lr = LinearRegression(X, y)
iters = 1500  # no. of iterations in Gradient Descent
alpha = 0.01  # learning rate

# visualizing the result
costs = lr.compute_optimal_theta(iters, alpha)
plt.plot([i for i in range(1, iters + 1)], costs)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost minimization - univariate')
plt.show()  # how gradient minimized the cost

print('Minimized cost: ' + str(lr.cost))
print('Optimal parameters: ')
print(lr.theta)

predicted_values = lr.X @ lr.theta
plt.scatter(data[0], data[1])
plt.plot(data[0], predicted_values, 'r')
plt.xlabel('Population of a City in 10000s')
plt.ylabel('Profit in $10000s')
plt.title('Optimal line')
plt.show()



# ----MULTIVARIATE DATA-----
input('PRESS ENTER TO CONTINUE \n')
print('----------------------')
print('-----MULTIVARIATE-----')
print('----------------------')

data_multi = pd.read_csv('data2.txt', header=None)
data_multi.columns = ['Size', 'No. of bedrooms', 'Price of the house']
print('First 5 examples from data2.txt')
print(data_multi.head())

data_multi.columns = [0,
                      1,
                      2]  # changing columns' headers because plt.scatter doesnt see values propely if column's name is a string

# visualizing the data
input('PRESS ENTER TO APPLY LINEAR REGRESSION \n')
X_multi = data_multi.iloc[:, 0:2].to_numpy(copy=True)
y_multi = data_multi.iloc[:, 2].to_numpy(copy=True).reshape((data_multi[1].shape[0], 1))

lr_multi = LinearRegression(X_multi, y_multi, normalize=True)  # necessary, otherwise it will throw an overflow
iters_multi = 400
alpha_multi = 0.01
costs_multi = lr_multi.compute_optimal_theta(iters_multi, alpha_multi)
plt.plot([i for i in range(1, iters_multi + 1)], costs_multi)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost minimization - multivariate')
plt.show()  # how gradient minimized the cost


print('Minimized cost: ' + str(lr_multi.cost))
print('Optimal parameters: ')
print(lr_multi.theta)

