import numpy as np
np.set_printoptions(precision = 6, suppress = True)
import csv
from sys import argv
# import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, Adam
from keras.utils import np_utils, plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import optimizers
SHAPE = 48
CATEGORY = 7
READ_FROM_NPZ = 1
DATA_DIR = './data/'
MODEL_DIR = './model'
#DATA_DIR = '/home/mirlab/kevin/ML_HW3/data/'

def read_train(filename):

	X, Y = [], []
	with open(filename, 'r', encoding='big5') as f:
		count = 0
		for line in list(csv.reader(f))[1:]:
			Y.append( float(line[0]) )
			X.append( [float(x) for x in line[1].split()] )
			count += 1
			print('\rX_train: ' + repr(count), end='', flush=True)
		print('', flush=True)

	return np.array(X), np_utils.to_categorical(Y, CATEGORY)


def main():
	print('============================================================')
	X, Y = [], []

	if READ_FROM_NPZ:
		print('Read from npz')
		data = np.load(DATA_DIR + '/data/data.npz')
		X = data['arr_0']
		Y = data['arr_1']
	else:
		print('Read train data')
		X, Y = read_train(DATA_DIR + argv[1])

	print('Reshape data')
	X = X/255
	X = X.reshape(X.shape[0], SHAPE, SHAPE, 1)

	print('============================================================')
	print('Construct model')
	model = Sequential()
	model.add(Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(MaxPooling2D((2, 2)))#Dimension reduction

	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(BatchNormalization(axis=-1))
	model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
	model.add(MaxPooling2D((2, 2)))

	model.add(Flatten())
	model.add(Dense(units = 256, activation='relu'))#fully connect
	model.add(Dense(units = 128, activation='relu'))
	model.add(Dense(units = 64, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(units = 7, activation='softmax'))
	model.summary()

	print('Compile model')
	# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
	adam = optimizers.Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	print('============================================================')
	VAL = 2400
	BATCH = 128
	EPOCHS = 40

	print('Train with raw data')
	es = EarlyStopping(monitor='val_acc', patience=3, verbose=1, mode='auto')
	history = model.fit(X, Y, batch_size=BATCH, epochs=EPOCHS, verbose=1, validation_split=0.1, callbacks=[es],shuffle = True)

	# print('============================================================')
	# print('show history')
	# # list all data in history
	# print(history.history.keys())
	# # summarize history for accuracy
	# plt.plot(history.history['acc'])
	# plt.plot(history.history['val_acc'])
	# plt.title('model accuracy')
	# plt.ylabel('accuracy')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# # plt.show()
	# fig = plt.gcf()
	# fig.savefig('accuracy.png',dpi =100)
	# # summarize history for loss
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	# plt.title('model loss')
	# plt.ylabel('loss')
	# plt.xlabel('epoch')
	# plt.legend(['train', 'test'], loc='upper left')
	# # plt.show()
	# fig1 = plt.gcf()	
	# fig1.savefig('loss.png')


	print('============================================================')
	print('Evaluate train')
	score = model.evaluate(X, Y)
	score = '{:.6f}'.format(score[1])
	print('Train accuracy (all):', score)

	print('============================================================')
	print('Save model')
	model.save(MODEL_DIR + '/' + score + '.h5')

if __name__ == '__main__':
	main()
