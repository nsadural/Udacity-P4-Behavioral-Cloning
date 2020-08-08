### Behavioral Cloning Project
##  Nikko Sadural
#   Create and train deep learning model and save a trained output model 'model.h5' for autonomous driving mode.
#   Script requires tensorflow==1.4, tensorflow-gpu==1.4, keras==2.1.3

import csv
import cv2
import numpy as np
import pandas as pd
from scipy import ndimage
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, GlobalAveragePooling2D
from keras.applications.inception_v3 import InceptionV3

## Read in .csv file with driving log data and append to list of lines
lines = []
with open('./mydata/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

## Read in .jpg image files and corresponding steering angle and append to list of images and measurements
images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './mydata/IMG/' + filename
    image = ndimage.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
    ## Augment data by flipping image along vertical axis and opposite steering measurement
    image_flip = np.fliplr(image)
    measurement_flip = -measurement
    images.append(image_flip)
    measurements.append(measurement_flip)

## Convert image data and labels to numpy arrays and shuffle
X_train = np.array(images)
y_train = np.array(measurements)
X_train, y_train = shuffle(X_train, y_train)

## Count number of unique steering angles from label data
df = pd.DataFrame(y_train)
n_classes = (df.nunique())[0]

## Flags for training model
freeze_flag = False                                                              # 'True' to freeze layers for pre-trained weights
weights_flag = 'imagenet'                                                       # use pre-trained weights from ImageNet images
preprocess_flag = True                                                          # 'True' for ImageNet pre-trained weights

## Create Inception model
input_size_y = 160
input_size_x = 320
color_channels = 3
inception = InceptionV3(weights=weights_flag, include_top=False, input_shape=(input_size_y, input_size_x, color_channels))

## Freeze pre-trained weights
if freeze_flag == True:
    for layer in inception.layers:
        layer.trainable = False
        
## Create normalizing input lambda layer
model = Sequential()
model.add(Lambda(lambda x: (x/255.0) - 0.5, input_shape=(input_size_y, input_size_x, color_channels)))
model.add(inception)
model.add(GlobalAveragePooling2D())
model.add(Dense(512, activation='relu'))
model.add(Dense(1))

## Compile model
model.compile(loss='mse', optimizer='adam')
        
## View inception model layers and visualize data
model.summary()
print()
print("Unique steering angles:", n_classes)

## Train and save model
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=4)
model.save('model.h5')