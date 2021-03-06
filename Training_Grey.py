import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model as M
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
import cv2
import os
from matplotlib import pyplot as plt
import time as t
import pickle
from keras.utils import to_categorical


# functions
# function to convert the time into something readable
def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# function to load .pickle files
def load_data(file_name):
    print(f'Loading {file_name}...')
    file = pickle.load(open(file_name, 'rb'))
    return file


# quick function to show the image
def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# reshapes the images to the right size
def reshape_data(X, y):
    print(f"Reshaping data...")
    X = np.array(X)     # ensuring that lists are instead arrays
    X = X / 255
    # triple_channel = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    y = np.array(y)
    y = to_categorical(y)
    # print(f"triple_channel.shape(): {triple_channel.shape}, y.shape(): {y.shape}")
    return X, y


# function to build the network
def build_network():
    mobile = keras.applications.mobilenet.MobileNet()
    x = mobile.layers[-6].output
    predictions = Dense(7, activation='softmax')(x)
    model = M(inputs=mobile.input, outputs=predictions)
    # makes only the last 5 layers trainable
    for layer in model.layers[:-23]:
        layer.trainable = False
    model.compile(Adam(lr=.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


# function to train the model
def train_model(model, imgs, labels):
    print("Training model...")
    model.fit(imgs, labels, epochs=NUM_EPOCHS, validation_split=0.1, batch_size=BATCH_SIZE)
    return model


# function to build a working model
def build_Mark_4_40(X):
    print("Building Mark 4.40...")
    model = Sequential()
    # First Layer
    model.add(
        Conv2D(512, kernel_size=(4, 4), activation='relu', input_shape=(X.shape[1:]), padding='same', strides=(2, 2)))
    # Second Layer
    model.add(Conv2D(256, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Third Layer
    model.add(Conv2D(128, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Fourth Layer
    model.add(Conv2D(128, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Fifth Layer
    model.add(Conv2D(256, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Sixth Layer
    model.add(Conv2D(512, kernel_size=(4, 4), activation='relu', padding='same', strides=(2, 2)))
    # Final Layer
    model.add(Flatten())
    model.add(Dense(7, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adamax',
                  metrics=['accuracy'])
    # All Done
    model.summary()
    return model


# global variables
CATAGORIES = ['Class (1)', 'Class (2)', 'Class (3)', 'Class (4)', 'Class (5)', 'Class (6)', 'Class (7)']
DATA = []
TESTING_DATA = []
IMG_SIZE = 224
NUM_EPOCHS = 4
BATCH_SIZE = 20


# Code to run
start_time = t.time()
print("Starting...")

# load in data
training_images = load_data('Images.pickle')
training_labels = load_data('Labels.pickle')
testing_images = load_data('Testing_Images.pickle')
testing_labels = load_data('Testing_Labels.pickle')

# reshape the data
training_images, training_labels = reshape_data(training_images, training_labels)
testing_images, testing_labels = reshape_data(testing_images, testing_labels)

# build and train the model
our_model = build_Mark_4_40(training_images)
trained_model = train_model(our_model, training_images, training_labels)

# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# evaluate the model
loss, acc = trained_model.evaluate(testing_images, testing_labels, batch_size=BATCH_SIZE, use_multiprocessing='True')
acc = round(acc * 100, 2)
print(f'accuracy: {acc}%')

# final message
print(f"Finished in: {total_time}")
print('Success!')