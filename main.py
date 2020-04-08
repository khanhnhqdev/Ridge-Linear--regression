import numpy as np 
from util import *
import ridge_regression

# load data
read = np.loadtxt('data.txt')
X = read[:, :-1]
Y = read[:, -1]
X = normalize_and_add_one(X)

# split train, test
X_train = X[:50]
Y_train = Y[:50]
X_test = X[50:]
Y_test = Y[50:]

# Find the best lamda
ridge_regression = ridge_regression.ridge_regression()
best_lamda = ridge_regression.get_the_best_lamda(X_train, Y_train)
print('best_lamda: ' , best_lamda)

# predict
w = ridge_regression.fit(X_train, Y_train, lamda = best_lamda)
print(w)
Y_predict = ridge_regression.predict(w, X_test)
print(Y_test)
print(Y_predict)
print(ridge_regression.compute_RSS(Y_test, Y_predict))


