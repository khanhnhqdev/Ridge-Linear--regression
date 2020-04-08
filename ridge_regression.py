import numpy as np
from numpy.linalg import inv

class ridge_regression:
	def __init__(self):
		return 

	def fit(self, X_train, Y_train, lamda):
		assert X_train.shape[0] == Y_train.shape[0] and len(X_train.shape) == 2
		w = inv(X_train.transpose().dot(X_train) + lamda * np.eye(X_train.shape[1])).dot(X_train.transpose()).dot(Y_train)
		return w

	def fit_gradient(self, X_train, Y_train, lamda, learning_rate, epochs = 100, batch_size = 128):
		w = np.random.randn(X_train.shape[1])
		loss = 1e8
		for epoch in range(epochs):
			arr = np.array(range(X_train.shape[0]))
			np.random.shuffle(arr)
			X_train = X_train[arr]
			Y_train = Y_train[arr]
			num_mini_batch = int(np.ceil(X_train.shape[0] / batch_size))
			for i in range(num_mini_batch):
				ind = i * batch_size
				X_train_tmp = X_train[ind : ind + batch_size]
				Y_train_tmp = Y_train[ind : ind + batch_size]
				grad = (X_train_tmp.transpose().dot(X_train_tmp) + lamda * np.eye(X_train_tmp.shape[1])).dot(w) - X_train_tmp.transpose().dot(Y_train_tmp)
				w = w - learning_rate * grad
			new_loss = self.compute_RSS(Y_train, self.predict(w, X_train))
			if(np.abs(loss - new_loss) < 1e-5):
				break
			loss = new_loss
		return w

	def predict(self, W, X_test):
   		X_test = np.array(X_test)
   		Y_test = X_test.dot(W)
   		return Y_test

	def compute_RSS(self, Y_true, Y_predict):
		loss = 1. / Y_true.shape[0] * (np.sum((Y_true - Y_predict) ** 2))
		return loss
	
	def get_the_best_lamda(self, X_train, Y_train):
		def cross_validation(num_fold, lamda):
			row_id = np.array(range(X_train.shape[0]))
			valid_ids = np.split(row_id[:len(row_id) - len(row_id) % num_fold], num_fold)	
			valid_ids[-1] = np.append(valid_ids[-1], row_id[len(row_id) - len(row_id) % num_fold :])
			train_ids = np.array([[k for k in range(X_train.shape[0]) if k not in valid_ids[i]] for i in range(num_fold)])
			# print(train_ids)
			aver_RSS = 0
			for i in range(num_fold):
				# print(train_ids[i])
				# print(valid_ids[i])
				train_X = X_train[train_ids[i]]
				train_Y = Y_train[train_ids[i]]
				valid_X = X_train[valid_ids[i]]
				valid_Y = Y_train[valid_ids[i]]
				w = self.fit(train_X, train_Y, lamda)
				valid_predict = self.predict(w, valid_X)
				aver_RSS += self.compute_RSS(valid_Y, valid_predict)
			return aver_RSS / num_fold

		def range_scan(best_lamda, minimum_RSS, lamda_range):
			for lamda in lamda_range:
				tmp_RSS = cross_validation(num_fold = 4, lamda = lamda)
				if tmp_RSS < minimum_RSS:
					best_lamda = lamda
					minimum_RSS = tmp_RSS
			return best_lamda, minimum_RSS

		best_lamda, minimum_RSS = range_scan(best_lamda = 0, minimum_RSS = 1000 ** 2, lamda_range = range(50))
		lamda_range = [k * 1 / 1000 for k in range(max(0, (best_lamda - 1) * 1000), (best_lamda + 1) * 1000, 1)]
		best_lamda, minimum_RSS = range_scan(best_lamda = best_lamda, minimum_RSS = minimum_RSS, lamda_range = lamda_range)
		return best_lamda

