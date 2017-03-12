import cv2
import os
import json
import csv
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import gen_batches, shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Input, Activation, Convolution2D, Flatten, Dropout, Cropping2D, Lambda, ELU



#----------------------------------------------------------------------------------------------
#read in samples info
samples = []
csvfile = pd.read_csv('./data/driving_log.csv')
for row in csvfile.iterrows():
    index, data = row
    samples.append(data.tolist())

#split data
train_samples, validation_samples = train_test_split(samples, test_size=0.05)


#----------------------------------------------------------------------------------------------
#generator
def generator(samples, batch_size):
    num_samples = len(samples)
    while 1: # Continuous loop
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []

            #reading and augmentation
            for batch_sample in batch_samples:
                center_name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(center_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                measurements.append(center_angle)
                images.append(cv2.flip(center_image,1))
                measurements.append(center_angle*-1.0)

                #left and right
                left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)
                left_angle = float(batch_sample[3]) + 0.5
                images.append(left_image)
                measurements.append(left_angle)
                images.append(cv2.flip(left_image,1))
                measurements.append(left_angle*-1.0)

                right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)
                right_angle = float(batch_sample[3]) - 0.5
                images.append(right_image)
                measurements.append(right_angle)
                images.append(cv2.flip(right_image,1))
                measurements.append(right_angle*-1.0)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=16)
validation_generator = generator(validation_samples, batch_size=16)


#----------------------------------------------------------------------------------------------
#exploratory analysis
def exploratory_analysis():
    
    #get some images
    images_subset = next(train_generator)[0]

    #print stats
    print("Training samples:", len(train_samples))
    print("Validation samples:", len(validation_samples))
    print("Image width:", len(images_subset[0]))
    print("Image height:", len(images_subset[0][0]))
    print("Image channels:", len(images_subset[0][0][0]))

    #plot 9 images
    rand_indices = np.random.randint(0, len(images_subset), 9)
    rand_images = images_subset[rand_indices]
    fig, axes = plt.subplots(nrows=3,ncols=3, figsize=(9,9))
    axes = axes.ravel()
    for ax, img in zip(axes, rand_images):
        ax.imshow(img)
        ax.axis('off')
    plt.show()

#exploratory_analysis()



#----------------------------------------------------------------------------------------------
#model
def model():
    #define model
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((50,25),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(36, 5, 5, subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(48, 5, 5, subsample=(2,2)))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3))
    model.add(ELU())
    model.add(Convolution2D(64, 3, 3))
    model.add(ELU())
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(50))
    model.add(Dropout(0.5))
    model.add(ELU())
    model.add(Dense(10))
    model.add(Dense(1))

    model.summary()

    model.compile(loss='mean_squared_error', optimizer='adam')


    #train
    history = model.fit_generator(train_generator, 
                                  samples_per_epoch = len(train_samples) * 6,
                                  nb_epoch = 1, 
                                  validation_data = (validation_generator),
                                  nb_val_samples = len(validation_samples) * 6,
                                  verbose = 1)                                
                               


    #save file
    model.save('model.h5')    

model()










