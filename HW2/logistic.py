import numpy as np
import csv
import math
from sys import argv

NUM_FEATURES = 106
ITERATION = 100
ADD_FEATURES = 1

ETA = 1e-4
ADAGRAD = 0

NORMALIZE = 1
LAMBDA = 0.01
global w,b

#使用set_printoptionsgk設置输出的精度（三位）,不用科學符號表示
np.set_printoptions(precision = 3, suppress = True)
#不會出現RuntimeWarning: invalid value encountered,因爲數组中存在0元素
np.seterr(divide='ignore', invalid='ignore')

def read_train(n):
	train = []
	
	with open(argv[n], 'r') as f:
		for line in list(csv.reader(f))[1:]:
			#line = [float(x) for x in line]
			train.append([float(x) for x in line])

	return np.array(train), len(train)
	

def read_test(n):
	test = []
	
	with open(argv[n], 'r') as f:
		for line in list(csv.reader(f))[1:]:
			#line = [float(x) for x in line]
			test.append([float(x) for x in line])
			
	return np.array(test), len(test)

def add_feature(X):
	global NUM_FEATURES
	X1 = X[:, [0]] ** 2
	X2 = X[:, [0]] * X[:, [5]]
	X3 = X[:, [0]] - X[:, [5]]
	NUM_FEATURES += 3
	index = list(range(106, NUM_FEATURES))
	return np.concatenate((X, X1, X2, X3), axis = 1), index


def training(X, Y, X_len, w, b):
	size = float(X_len)
	gw_all = np.zeros(w.shape)
	gb_all = 0.0
	for i in range(ITERATION):	
		# Ein
		correct = 0.0
		for j in range(X_len):
			z = np.dot(X[j], w) + b
			predict = 1.0 if z >= 0.0 else 0.0
			correct += 1 if predict == Y[j]	else 0
		print(i, "Correct Rate:", correct / size)

		gw = np.zeros(w.shape)
		gb = 0.0
		for j in range(X_len):	
			z = (np.dot(X[j], w) + b)
			gw += (sigmoid(z) - Y[j]) * X[j]
			gb += (sigmoid(z) - Y[j])

		if ADAGRAD:
			gw_all += (gw/size) ** 2
			gb_all += (gb/size) ** 2
			w_ = ETA * gw / np.sqrt(gw_all)
			b_ = ETA * gb / np.sqrt(gb_all)
		else:
			w_ = ETA * gw
			b_ = ETA * gb

		w -= w_
		b -= b_

def sigmoid(z):
	if not NORMALIZE: # z fix
		z = z * 1e-9 + 1
	return 1. / (1. + np.exp(-z))

def output(n, result):
	with open(argv[n], "w") as f:
		f.write("id,label\n")
		for i in range(len(result)):
			f.write(repr(i + 1) + "," + repr(result[i]) + "\n")

def main():
	X_train, X_len = read_train(3)
	Y_train, Y_len = read_test(4)
	print("X length:", X_len)
	print("Y length:", Y_len)
	#print(np.shape(X_train[1]))
	global w, b

	#add features
	other_index = []
	if ADD_FEATURES:
		X_train, other_index = add_feature(X_train)

	w = np.array([1e-4 for x in range(NUM_FEATURES)])#做一個 NUM_FEATURES*1 長的w
	b = 0.0

	# train normalization
	norm_index = [0, 1, 3, 4, 5] + other_index
	if NORMALIZE:
		np.random.seed(100)
		w = 0.001 * (np.random.random(NUM_FEATURES) * 2 - 1)
		b = 0.001 * (np.random.random() * 2 - 1)
		for i in norm_index:
			mu = np.mean(X_train[:, i], axis = 0)
			sig = np.std(X_train[:, i], axis = 0)
			X_train[:, i] = (X_train[:, i] - mu) / sig


	#print('w shape:',np.shape(w))
	#train
	training(X_train, Y_train, X_len, w, b)

	# test normalization
	X_test, X_test_len = read_train(5)
	if ADD_FEATURES:
		X_test, other_index = add_feature(X_test)
	for i in norm_index:
		mu = np.mean(X_test[:, i], axis = 0)
		sig = np.std(X_test[:, i], axis = 0)
		X_test[:, i] = (X_test[:, i] - mu) / sig

	# test
	result = []
	for i in range(X_test_len):
		z = np.dot(w, X_test[i]) + b
		p = sigmoid(z)
		if p >= 0.5:
			result += [1]
		else:
			result += [0]

	output(6, result)	



if __name__ == "__main__":
	main()