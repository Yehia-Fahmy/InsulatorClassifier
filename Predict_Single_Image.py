# this file should be able to predict a single image using the tflite model and print the result to the console

import numpy as np
from tensorflow import keras
from matplotlib import pyplot as plt
import time as t
import pickle
import tensorflow as tf


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