from sys import argv
from random import shuffle
import numpy as np
import csv

global train, test
train = [[] for _ in range(18)]
test = []

#training data
with open(argv[1], 'rb') as f:
	data = f.read().splitlines()
		
	i = 0
	for line in data[1:]: # trim the first data which is header
		line = [x.replace("\'", "") for x in str(line).split(',')[3:]]

		if i % 18 == 10:
			line = [x.replace("NR", "0") for x in line]

		line = [float(x) for x in line]
		train[i % 18] += line
			
		i += 1



#testing data		
with open(argv[2], 'rb') as f:
	data = f.read().splitlines()
	i = 0
	for line in data: # trim the first data which is header
		line = [x.replace("\'", "") for x in str(line).split(',')[2:]]

		if i % 18 == 10:
			line = [x.replace("NR", "0") for x in line]

		line = [float(x) for x in line]
		test.append(line)

		i += 1



# default parameters
global ITERATION, ETA, VALIDATION, SGD, BATCH, PERIOD, MAX_TIME,FEATURE, NUM_FEATURE, w, b
ITERATION = 4500
ETA = 1.25e-8
VALIDATION = 0 # validation size
DATA_SIZE = 1

SGD=0
BATCH = 500

PERIOD = 7
MAX_TIME = 0

#FEATURE = range(18)
FEATURE = [7, 9, 12] # selected feature
NUM_FEATURE = len(FEATURE)

# default starting point
w = np.array([[0.01] * PERIOD] * NUM_FEATURE) # training maxtrix
b = 1.0 # bias

# set numpy print option
np.set_printoptions(precision = 6, suppress = True)

def print_message():
	print("\nLinear Regression")
	print("iteration =", ITERATION)
	print("eta =", ETA)
	print("max time=", MAX_TIME);
	print("validation =", VALIDATION)
	print("selected features =", FEATURE)
	print("period =", PERIOD)
	print("w =\n", w)
	print("b =", b)


def filter(data, start, period, selected_features):
	result = []
	for f in selected_features:
		result += [data[f][start : start + period]]

	return result

def training():
	global train, test, w, b
	all_Ein=[]

	for i in range(ITERATION):
		if i % 100 ==0:
			print("now is iteration: ",i )

		b_grad = 0.0
		w_grad = np.zeros([NUM_FEATURE, PERIOD])
		iter_Ein=[]
		for start in range(MAX_TIME - PERIOD -1)[VALIDATION:]:

			X = np.array( filter(train, start, PERIOD, FEATURE))
			yh = train[9][start + PERIOD]

			b_grad = b_grad - 2.0 * (yh - predict(X, w, b)) * 1.0
			w_grad = w_grad - 2.0 * (yh - predict(X, w, b)) * X

			Ein = (yh - predict(X, w, b)) ** 2
			iter_Ein.append(Ein)

		# update parameters
		w = w - ETA * w_grad
		b = b - ETA * b_grad

		all_Ein.append(iter_Ein)

		if i % 10 == 0:
			print("current Ein =", np.sqrt(np.mean(all_Ein[-10:])))

	return all_Ein		

def predict(X, w, b):
	Y = np.sum(X * w) + b
	return Y


def testing():
	with open(argv[3],'w')as f:
		f.write("id,value\n")
		for d in range(240):

			data = np.array( filter( test[ d * 18 : (d + 1) * 18], 9 - PERIOD, PERIOD, FEATURE))
			dot_result = int(predict(data, w, b))
			f.write("id_" + repr(d) + "," + repr(dot_result) + "\n")



def main():
	global train, test, MAX_TIME
	global ITERATION, ETA, VALIDATION, SGD, BATCH, PERIOD, MAX_TIME,FEATURE, NUM_FEATURE, w, b
	MAX_TIME = int(len(train[0]) * (DATA_SIZE))
	if len(argv) > 4: # if there is parameter file, load it
		ITERATION, ETA, VALIDATION, SGD, BATCH, PERIOD, FEATURE, NUM_FEATURE, w, b = read_parameter(argv[4])
		predict_test()
		return

	print_message()
	all_Ein = training()
	print_message()

	average_Ein = np.sqrt(np.mean(all_Ein))
	final_Ein =  np.sqrt(np.mean(all_Ein[-10:]))
	Ev = validation()
	print("average Ein =", average_Ein, "\nfinal Ein =", final_Ein, "\nEvalid =", Ev)
	
	testing()
	output_parameter(average_Ein, final_Ein, Ev)

def validation():
	Evalid = []
	for start in range(MAX_TIME - PERIOD - 1)[:VALIDATION]:

		X = np.array( filter(train, start, PERIOD, FEATURE) )
		yh = train[9][start + PERIOD]

		Ev = (yh - predict(X, w, b)) ** 2
		Evalid.append(Ev)

	return np.sqrt(np.mean(Evalid)) if VALIDATION else 0

def output_parameter(average_Ein, final_Ein, Ev):	
	filename = "LR"
	filename += "_" + repr(ITERATION) + "i"
	filename += "_" + repr(ETA)
	filename += "_" + repr(NUM_FEATURE) + "f"
	filename += "_" + "{:.3f}".format(final_Ein) + "Ein"

	with open( filename + ".csv", "w") as f:
		
		f.write("iteration," + repr(ITERATION) + "\n")
		f.write("eta," + repr(ETA) + "\n")
		f.write("period," + repr(PERIOD) + "\n")
		f.write("sgd," + repr(SGD) + "\n")
		f.write("batch," + repr(BATCH) + "\n")
		
		f.write("feature,")
		for r in FEATURE:
			f.write(repr(r) + ",")
		f.write("\n")
		
		f.write("average Ein," + repr(average_Ein) + "\n")
		f.write("final Ein," + repr(final_Ein) + "\n")
		f.write("validation," + repr(VALIDATION) + "\n")
		f.write("Ev," + repr(Ev) + "\n")
		f.write("b," + repr(b) + "\n")

		for row in w:
			f.write("w,")
			for r in row:
				f.write(repr(r) + ",")
			f.write("\n")

if __name__ == "__main__":
	main()	