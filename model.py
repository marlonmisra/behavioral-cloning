import cv2
import os
import json
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import gen_batches
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Convolution2D, Flatten, Dropout, Cropping2D, Lambda




#-----------------------------------------------------------------------------------------------
#data reading

#read the features
def read_features(folder):
	features =[]
	for filename in os.listdir(folder):
		#only include center images because no target info ofor other ones
		if filename.startswith('center_'):
			img = cv2.imread(os.path.join(folder, filename))
			features.append(img)
	return np.asarray(features)

#read the targets
def read_targets(file):
	targets = pd.read_csv(file)
	targets = np.array(targets)[:,3]    #only include the steering angle column
	return targets


#set up data dictionary
def read_data(features_folder, targets_file):

	data =  {
	'features': None,
	'targets': None
	}

	data['features'] = read_features(features_folder)
	data['targets'] = read_targets(targets_file)

	print("Features shape:", data['features'].shape)
	print("Targets shape:", data['targets'].shape)

	print("Finished reading data.")
	return data


data = read_data('IMG', 'driving_log.csv')
X = data['features']
y = data['targets']
n_samples = data['features'].shape[0]
height = data['features'].shape[1]
length = data['features'].shape[2]
channels = data['features'].shape[3]


#----------------------------------------------------------------------------------------------
#exploratory analysis

#plot 9 random images
def make_plots():
	rand_indices = np.random.randint(0, n_samples, 9)
	rand_images = X[rand_indices]
	fig, axes = plt.subplots(nrows=3,ncols=3, figsize=(9,9))
	axes = axes.ravel()
	for ax, img in zip(axes,rand_images):
		ax.imshow(img)
		ax.axis('off')
	plt.show()

#make_plots()


#----------------------------------------------------------------------------------------------
#preprocessing

#slight blur for better generalizability
def blur(image):
    kernel_size = 3
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


#avoid numerical stability issues
def normalize(image):
	return image/255

#apply feature processing functions
def process(features):
	features_processed = np.empty((n_samples, height, length, channels))
	for i in range(0, n_samples):
		features_processed[i] = normalize(blur(features[i]))
	print("Finished processing data.")
	return features_processed

X_processed = process(X)


#----------------------------------------------------------------------------------------------
#training and validation split

#split function
def split_data():
	X_train, X_valid, y_train, y_valid = train_test_split(X_processed, y, test_size = 0.1, random_state = 0)
	print("Training samples:", len(X_train))
	print("Validation samples:", len(X_valid))

	print("Finished splitting data.")
	return X_train, X_valid, y_train, y_valid

X_train, X_valid, y_train, y_valid = split_data()


#----------------------------------------------------------------------------------------------
#model

def model():
	#define model
	model = Sequential()
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(height, length, channels)))
	#model.add(Lambda(lambda x: (x / 255.0) - 0.5))
	model.add(Convolution2D(16, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Activation('relu'))
	model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
	model.add(Flatten())
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(200))
	model.add(Activation('relu'))
	model.add(Dense(1))

	model.summary()

	model.compile(loss='mean_squared_error', optimizer='adam')



	#train
	history = model.fit(X_train, y_train, batch_size = 64, nb_epoch = 10, validation_data = (X_valid, y_valid))

	#save files
	model.save_weights("model.h5", True)
	with open('model.json', 'w') as outfile:
	    json.dump(model.to_json(), outfile)

model()

