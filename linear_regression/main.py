import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression.lin_reg import LinearRegression

# first we will execute an univariate linear regression

# data1.txt is a data taken from Andrew Ng's Machine Learning course - Linear Regression
# in data1.txt, first column is the population of a city (in 10000s) and the second column is a profit of a food truck (int $10000s)


# showing the data
data = pd.read_csv('data1.txt', header=None)
data.columns = ['Population', 'Profit']
print(data.head())

# visualising the data
data.columns = [0,
                1]  # changing columns' headers because plt.scatter doesnt see values propely if column's name is a string
plt.scatter(data[0], data[1])
plt.xlabel('Population of a City in 10000s')
plt.ylabel('Profit in $10000s')
plt.title('Training data')
plt.show()

X = data[0].to_numpy(copy=True).reshape((data[0].shape[0], 1))
y = data[1].to_numpy(copy=True).reshape((data[1].shape[0], 1))

lr = LinearRegression(X, y)
iters = 1500  # no. of iterations in Gradient Descent
alpha = 0.01  # learning rate

# visualizing the result
costs = lr.compute_optimal_theta(iters, alpha)
plt.plot([i for i in range(1, iters + 1)], costs)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost minimization')
plt.show()  # how gradient minimized the cost

print('Minimized cost: ' + str(lr.cost))
print('Optimal parameters: ')
print(lr.theta)

predicted_values = np.append(np.ones((X.shape[0], 1)), X, 1) @ lr.theta
plt.scatter(data[0], data[1])
plt.plot(data[0], predicted_values, 'r')
plt.xlabel('Population of a City in 10000s')
plt.ylabel('Profit in $10000s')
plt.title('Optimal line')
plt.show()
