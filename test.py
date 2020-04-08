import numpy as np 
from util import *
import ridge_regression

# load data
# read = np.loadtxt('data.txt')
# X = read[:, :-1]
# Y = read[:, -1]
# X = normalize_and_add_one(X)

# # split train, test
# X_train = X[:50]
# Y_train = Y[:50]
# X_test = X[50:]
# Y_test = Y[50:]

ridge_regression = ridge_regression.ridge_regression()

# test compute_RSS
# loss = ridge_regression.compute_RSS(np.array([1,2,3,4]), np.array([1,2,3,4]))
# print(loss)

# test predict
# pre = ridge_regression.predict(np.array([[1], [2], [3], [4]]), np.array([[1,2,3,4], [1,1,1,1]]))
# print(pre)

# test fit_gradient
# w1 = ridge_regression.fit(X_train, Y_train, lamda = 1)
# w2 = ridge_regression.fit_gradient(X_train, Y_train, lamda = 1, epochs = 10000, learning_rate = 0.001)

# print(w1 - w2)

# test predict
X = np.array([[1,3,4], [1,5,100], [1,7,8], [1,123,4]])
Y = np.array([[8],[106],[16],[128]])

best_lamda = ridge_regression.get_the_best_lamda(X, Y)
print('best_lamda: ' , best_lamda)

# predict
w = ridge_regression.fit(X, Y, lamda = 0)
y = ridge_regression.predict(w, X)
print(y)
print(ridge_regression.compute_RSS(y, Y))
