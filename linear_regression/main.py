import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression.lin_reg import LinearRegression
from linear_regression.lin_reg import simple_generate_data

# data1.txt and data2.txt are taken from Andrew Ng's Machine Learning course - Linear Regression
# I took data from Andrew Ng's course because correct answers are well known thus it's easy to test the code
# I also generated some data by myself to test if it works for more complex training data

# -----UNIVARIATE DATA-----
# in data1.txt, first column is the population of a city (in 10000s) and the second column is a profit of a food truck (int $10000s)
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

# example prediction
# Population of 35000
print('Predicted profit for a city with population of 35000: $' + str(round(lr.predict(np.array([[3.5]])) * 10000, 2)))
print('Predicted profit for a city with population of 70000: $' + str(round(lr.predict(np.array([[7.0]])) * 10000, 2)))

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

print('Predicted price of 1650 sq-ft, 3br house: $' + str(round(lr_multi.predict(np.array([[1650, 3]])), 2)))
print('Predicted price of 2000 sq-ft, 4br house: $' + str(round(lr_multi.predict(np.array([[2000, 4]])), 2)))

print('PRESS ENTER TO CONTINUE \n')
print('----------------------')
print('------POLYNOMIAL------')
print('----------------------')
simple_generate_data(3,[10,2,0.3],100)
data = pd.read_csv('generated_data.txt', header=None)

data.columns = ['X', 'Y']
data=data.sort_values(by=['X']) #will be useful later to plot the result correctly
print('First 5 sorted examples')
print(data.head())

data.columns = [0,
                1]  # changing columns' headers because plt.scatter doesnt see values propely if column's name is a stringhttps://towardsdatascience.com/polynomial-regression-bbe8b9d97491

plt.scatter(data[0], data[1])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Training data - polynomial')
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
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Optimal line')
plt.show()


print('On the graph we can see that simple linear function will not do the job. We need a polynomial')
print('Lets try to use a second degree polynomial.')
print('To achieve that, we need to add one more column of X squared')
input('PRESS ENTER TO APPLY POLYNOMIAL REGRESSION\n')
X=np.append(X ,np.square(X),1)
lr = LinearRegression(X, y,normalize=True)
iters = 1500  # no. of iterations in Gradient Descent
alpha = 0.01  # learning rate

# visualizing the result


costs = lr.compute_optimal_theta(iters, alpha)

plt.plot([i for i in range(1, iters + 1)], costs)
plt.xlabel('Number of iterations')
plt.ylabel('Cost')
plt.title('Cost minimization - polynomial')
plt.show()  # how gradient minimized the cost

print('Minimized cost: ' + str(lr.cost))
print('Optimal parameters: ')
print(lr.theta)

predicted_values = lr.X @ lr.theta
plt.scatter(data[0], data[1])
plt.plot(data[0], predicted_values, 'r')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Optimal line')
plt.show()

