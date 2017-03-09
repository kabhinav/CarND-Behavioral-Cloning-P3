import csv
import cv2
import os
import numpy as np
import sklearn
import tqdm

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


CORRECTION = 0.2

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
        
train_samples, validation_samples = train_test_split(rows, test_size=0.2)

def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                 sangles = steering_angles[','.join(batch_sample)]
                 for i in range(3):
                     filename = os.path.basename(batch_sample[i])
                     current_path = os.path.join('data', 'IMG', filename)
                     image = cv2.imread(current_path)
                     assert image is not None
                     images.append(image)
                     angles.append(sangles[i])
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples)
validation_generator = generator(validation_samples)


# model
model = Sequential()
model.add(Lambda(lambda x: (x/255.0)-0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# print model information
model.summary()

# train
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_samples)*3,
                    validation_data=validation_generator, nb_val_samples=len(validation_samples),
                    nb_epoch=3)
model.save('model.h5')
