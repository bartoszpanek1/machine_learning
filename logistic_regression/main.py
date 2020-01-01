import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression.lin_reg import LinearRegression

# data1.txt and data2.txt are taken from Andrew Ng's Machine Learning course - Logistic Regression
# I took data from Andrew Ng's course because correct answers are well known thus it's easy to test


data = pd.read_csv("data1.txt", header=None)
data.columns = ["Exam 1 score", "Exam 2 score", "Admitted"]
print(data.head())
data.columns = [0, 1,
                2]  # changing columns' headers because plt.scatter doesnt see values propely if column's name is a string

X = data.iloc[:, 0:2].to_numpy(copy=True)
y = data.iloc[:, 2].to_numpy(copy=True).reshape((data[2].shape[0], 1))
plt.scatter(X[(y == 1)[:, 0],0], X[(y == 1)[:, 0],1], c='blue', marker='s')
plt.scatter(X[(y == 0)[:, 0],0], X[(y == 0)[:, 0],1], c='yellow',
            marker='o')
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend(['Admitted','Not admitted'])
plt.show()
