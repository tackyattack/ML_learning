import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


path = os.getcwd() + '/data/ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
data.head()

print(data.describe())


# LEARNING AREA
# -------------------

# output = W1*F1 + W2*F2 + W3*F3 ...
# Wx is the weight for x feature
# Fx is feature x
def predict(features, weights):
    predictions = np.dot(features, weights)
    return predictions

# ----
# MSE (L2) is the cost cost function
# MSE measures the average squared distance between
# an observation's actual and predicted values.
# The output is a single value that represents a score
# for close we are (the error)

# Univariate:
# MSE = sum[(actual - predicted)^2]/N
# so you just find the difference, square it, sum, then divide by N (number of points)
# using the linear equation: y = mx + b
# MSE = sum[(y_i - (m*x_i+b))^2]/N    y_i is the actual value

# Multivariate:
# https://www.stat.cmu.edu/~cshalizi/mreg/15/lectures/13/lecture-13.pdf
# e := error function
# B := the coefficients (vector 2x1)
# y := output values (vector)
# x := features (nx2) matrix, where n is the number of features

# output = B0 + (B1)X   B0 is the bias term, B1 is the weights

# Question: what's with the weird
# Answer:
# you want 1's in the first column like so:
# x = [(1, x1), (1, x2), ... , (1, xn)]
# This is because when multipled by the weights you get:
# xB = [(B0 + B1*x1), (B0 + B1*x2), ... , (B0 + B1*xn)]
# this puts it in the correct form for getting the point predictions (found using the prediction function)
# --> Treats the intercept term as simply another "feature"

# error found by:
# e(B) = y - xB
# MSE(B) = sum[e(B_i)^2]/N
# using linear algebra: ((A.T)A = AA)
# MSE(B) = (e.T)e / N

def cost_function(features, targets, weights):
    N = len(targets)

    # get the output based on the features and their given weights
    predictions = predict(features, weights)

    # get the error by subtracting the targets from the prediction value
    error = (predictions - targets)

    # square the error
    sq_error = error**2

    # Return average squared error among predictions
    # note: divide by 2 to avoid doing it during the gradient descent (just makes the math easier)
    return 1.0 * sq_error.sum() / (2*N)

#----
# -- The gradient descent --

# First we need the gradient:
# This is the vector that will describe the slope of the cost function using our weights.
# Simply use the chain rule on the cost function to get:
# f'(W1) = -x1(y - (W1*x1 + W2*x2 + W3*x3))
# f'(W2) = -x2(y - (W1*x1 + W2*x2 + W3*x3))
# f'(W3) = -x3(y - (W1*x1 + W2*x2 + W3*x3))
# predictions: W1*x1 + W2*x2 + W3*x3
# targets: y


# more generally:
# gradient = -X(targets - predictions)
# gradient = - X.T * (targets - predictions)  <- has the correct shape so it can be multiplied
# notice how predictions, targets swap to bring the negative out front
# transpose X so you can multiply with the error matrix (has the correct shape)
# X: features matrix = [(x1, x2, x3, xn), (x1, x2, x3, xn), ...]

def gradientDescent2(X, weights, lr):
    # first get the predictions:
    predictions = predict(X, weights)

    # calculate the error
    error = targets - predictions

    # calculate the gradient
    gradient = np.dot(-X.T, error)

    # take the average
    gradient = gradient /= len(X)

    # multiply the the gradient by the learning rate
    gradient *= lr

    # subtract from the weights to minimize cost
    # you're taking a step down the gradient (-gradient)
    # how large your steps are is determined by the learning rate
    weights -= gradient

    return weights

# -------------------


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
