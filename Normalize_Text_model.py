import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD, Adam

import cv2
from os import listdir
from os.path import isfile, isdir, join
import numpy as np

index = 0
def image_normalize(image):
	global index
	img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	kernel = np.ones((1, 1), np.uint8)
	img = cv2.dilate(img, kernel, iterations = 1)
	img = cv2.erode(img, kernel, iterations=1)
	__, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
	h, w = img.shape
	# 假定圖片大小要變成 85*40
	ratio = w / h # 比例為 2.125
	if ratio <= 2.125:
		if h < w:
			x_dif = (2.125 * h) - w # 找出寬還差多少
			x_pad = int(x_dif/2)
			padding = np.zeros((h, x_pad), np.uint8)
			padding = padding + 255
			print("h: {}, w: {}, x_pad: {}, ratio: {}".format(h,w,x_pad, ratio))
			img = np.hstack((padding, img, padding))
		else:
			y_dif = (w / 2.125) - h # 找出高還差多少 padding 才能使圖片比例為 2.125
			y_pad = int(y_dif/2)
			print("h: {}, w: {}, y_pad: {}, ratio: {}".format(h,w,y_pad,ratio))
			padding = np.zeros((y_pad, w), np.uint8)
			padding = padding + 255
			img = np.vstack((padding, img, padding))
		"""
		if h < w:
			y_dif = (w / ratio) - h # 找出高還差多少 padding 才能使圖片比例為 2.125
			y_pad = int(y_dif/2)
			print("h: {}, w: {}, y_pad: {}, ratio: {}".format(h,w,y_pad,ratio))
			padding = np.zeros((y_pad, w), np.uint8)
			padding = padding + 255
			img = np.vstack((padding, img, padding))
		else:
			x_dif = (ratio * h) - w # 找出寬還差多少
			x_pad = int(x_dif/2)
			padding = np.zeros((h, x_pad), np.uint8)
			padding = padding + 255
			img = np.hstack((padding, img, padding))
		"""
	elif ratio > 2.125:
		"""
		x_dif = (ratio * h) - w # 找出寬還差多少
		x_pad = int(x_dif/2)
		print("h: {}, w: {}, x_pad: {}, ratio: {}".format(h,w,x_pad, ratio))
		padding = np.zeros((h, x_pad), np.uint8)
		img = np.hstack((padding, img, padding))
		"""
		y_dif = (w / 2.125) - h # 找出高還差多少 padding 才能使圖片比例為 2.125
		y_pad = int(y_dif/2)
		print("h: {}, w: {}, y_pad: {}, ratio: {}".format(h,w,y_pad,ratio))
		padding = np.zeros((y_pad, w), np.uint8)
		padding = padding + 255
		img = np.vstack((padding, img, padding))
		
	img = cv2.resize(img, (85,40),interpolation=cv2.INTER_CUBIC)
	#__, finish = cv2.threshold(img, 85, 255, cv2.THRESH_BINARY_INV)
	kernel = np.ones((1, 1), np.uint8)
	img = cv2.dilate(img, kernel, iterations = 1)
	img = cv2.erode(img, kernel, iterations=1)	
	#cv2.imshow("one", img)
	#cv2.waitKey(0)
	cv2.imwrite("one_or_two\\normallized-"+ str(index) +".jpg", img)
	index = index +1
	return img


def main():
	x_train = []
	y_train = []
	x_test = []
	y_test = []
	path_one = "C:\\Users\\PT\\liu\\one_or_two\\one"
	path_two = "C:\\Users\\PT\\liu\\one_or_two\\two"
	path_one_test = "C:\\Users\\PT\\liu\\one_or_two\\one_test"
	path_two_test = "C:\\Users\\PT\\liu\\one_or_two\\two_test"
	
	file = listdir(path_one)
	for f in file:
		full_path = join(path_one, f)
		img = cv2.imread(full_path)
		normal_img = image_normalize(img)
		normal_img = normal_img.astype('float32')
		x_train.append(normal_img)
		y = 0
		#y = np.array([y])
		y_train.append(y)
	
	file = listdir(path_two)
	for f in file:
		full_path = join(path_two, f)
		img = cv2.imread(full_path)
		normal_img = image_normalize(img)
		normal_img = normal_img.astype('float32')
		x_train.append(normal_img)
		y = 1
		#y = np.array([y])
		y_train.append(y)
	
	
	all_test_file = []
	all_test_label  =[]
	file = listdir(path_one_test)
	for f in file:
		full_path = join(path_one_test, f)
		img = cv2.imread(full_path)
		normal_img = image_normalize(img)
		normal_img = normal_img.astype('float32')
		x_test.append(normal_img)
		y = 0
		#y = np.array([y])
		y_test.append(y)
		all_test_file.append(f)
		all_test_label.append(y)
		
	file = listdir(path_two_test)
	for f in file:
		full_path = join(path_two_test, f)
		img = cv2.imread(full_path)
		normal_img = image_normalize(img)
		normal_img = normal_img.astype('float32')
		x_test.append(normal_img)
		y = 1
		#y = np.array([y])
		y_test.append(y)
		all_test_file.append(f)
		all_test_label.append(y)
		
	num_classes = 2
	img_rows = 40
	img_cols = 85
	batch_size = 32
	epochs = 20

	x_train = np.array(x_train)
	y_train = np.array(y_train)
	x_test = np.array(x_test)
	y_test = np.array(y_test)
	x_train /= 255
	x_test /= 255	

	# 根據模型的 image shape 調整通道位置
	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		#x_t = x_t.reshape(x_t.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		#x_t = x_t.reshape(x_t.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)
	
	# 分成 keras 的兩類
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)
	
	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	#model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(100, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))
	model.summary();

	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer='Adam',
				  metrics=['accuracy'])
	# Adam
	model.fit(x_train, y_train,
			  batch_size=batch_size,
			  epochs=epochs,
			  verbose=1)
	score = model.evaluate(x_test, y_test, batch_size=73)
	
	predictions = model.predict(x_test)
	for idx, pre in enumerate(predictions):
		i = np.argmax(pre)
		if i != all_test_label[idx]:
			print("the evaluate: ", pre)
			print("the index: ", idx)
			print("file: ", all_test_file[idx])
	
	file_name = str(score[1])
	model.save("SaveModel\\"+file_name + "_model.h5")

	model.save_weights("SaveModel\\"+file_name + "_weight.h5")
	print('Test loss:', score[0])
	print('Test accuracy:', score[1])

if __name__ == "__main__":
	main()