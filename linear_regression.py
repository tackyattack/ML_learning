import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path = os.path.normpath(os.getcwd() + os.sep + os.pardir + "/data/ex1data1.txt")
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()

print(data.describe())


#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))
#plt.show(block=True)

# every point in the training data can be written as
# the unknown function we are trying to approximate plus
# an error term which is drawn from a normal distribution of
# zero mean and unknown variance
# d(x_i) = f(x_i) + e(i)
# we want to minimize error (e), which is the least-square error
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))


# to make the pandas frame work, we need to insert column of 1s
# to make the matrix operations work properly

# append a ones column to the front of the data set
data.insert(0, 'Ones', 1)

# set X (training data) and y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

# convert to numpy matrices and create parameter matrix
X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

# what is the error with 0s? Should be pretty high since it hasn't been trained
print(computeCost(X, y, theta))

# LEARN THIS
# https://en.wikipedia.org/wiki/Gradient_descent
# https://www.youtube.com/watch?v=sDv4f4s2SB8
# https://www.youtube.com/watch?v=IHZwWFHWa-w&t=1097s
# use the exercies folder to learn how this works

# https://ml-cheatsheet.readthedocs.io/en/latest/linear_regression.html
# gives a nice explanation
# take a look at the version near the bottom that uses linear algebra
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta) # for tracking the cost over
                                           # each iteration

    return theta, cost


# initialize variables for learning rate and iterations
alpha = 0.01 # learning rate
iters = 1000 # iterations

# perform gradient descent to "fit" the model parameters
g, cost = gradientDescent(X, y, theta, alpha, iters)

# how is the error now?
print(computeCost(X, y, g))

print(g)


# plot the solution
x = np.linspace(data.Population.min(), data.Population.max(), 100)
f = g[0, 0] + (g[0, 1] * x)

fig, ax = plt.subplots(figsize=(12,8))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Population, data.Profit, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
plt.show()

# plot how the error changes with each iteration
# this is a convex optimization problem
# if you were to plut the entire solution space, it
# would create a bowl shape representing the optimal solution
fig, ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
plt.show(block=True)
