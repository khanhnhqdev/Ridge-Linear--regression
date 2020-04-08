import numpy as np 
import pandas as pd 

def normalize_and_add_one(X):
    X = np.array(X)
    normalize = {}
    # print(np.array(np.max(X[:, col_id]) for col_id in range(X.shape[1])))
    normalize['min'] = np.array([np.max(X[:, col_id]) for col_id in range(0,X.shape[1])])
    normalize['max'] = np.array([np.max(X[:, col_id]) for col_id in range(0,X.shape[1])]) 
    
    for row_id in range(0, X.shape[0]):
      X[row_id, :] = (X[row_id, :] - normalize['min']) / normalize['max']

    return np.column_stack((np.ones(X.shape[0]), X))