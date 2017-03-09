import csv
import cv2
import os
import numpy as np
import tqdm




CORRECTION = 0.1

# load data
rows = []
steering_angles = {}
with open('data/driving_log.csv') as driving_log:
    reader = csv.reader(driving_log)
    #next(reader)
    for row in reader:
        scenter = float(row[3])
        sleft = scenter + CORRECTION
        sright = scenter - CORRECTION
        skey = ','.join(row)
        steering_angles[skey] = (scenter, sleft, sright)
        rows.append(row)
        

# store data
images = []
measurements = []
print('... Reading training images')
for row in tqdm.tqdm(rows):
    sangles = steering_angles[','.join(row)]
    for i in range(3):
        filename = os.path.basename(row[i])
        current_path = os.path.join('data', 'IMG', filename)
        image = cv2.imread(current_path)
        assert image is not None
        images.append(image)
        measurements.append(sangles[i])

# data augmentation
augmented_images = []
augmented_measurements = []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(measurement*-1.0)
        
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# model
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.utils import np_utils


# layers
model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(700, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(350, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))

# print model information
model.summary()

# train
model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True)
model.save('model.h5')
