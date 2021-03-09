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


# load the tf lite model and allocate the tensors
interpreter1 = tf.lite.Interpreter(model_path="TF_Lite_Model.tflite")
interpreter2 = tf.lite.Interpreter(model_path="model_unquant.tflite")

interpreter1.allocate_tensors()
interpreter2.allocate_tensors()

# get input and output tensors
input_details1 = interpreter1.get_input_details()
output_details1 = interpreter1.get_output_details()
input_details2 = interpreter2.get_input_details()
output_details2 = interpreter2.get_output_details()

print('---input details---')
print(input_details1)
print(input_details2)

# test model on random input sample
input_shape1 = input_details1[0]['shape']
input_data1 = np.array(np.random.random_sample(input_shape1), dtype=np.float32)
interpreter1.set_tensor(input_details1[0]['index'], input_data1)

input_shape2 = input_details2[0]['shape']
input_data2 = np.array(np.random.random_sample(input_shape2), dtype=np.float32)
interpreter2.set_tensor(input_details2[0]['index'], input_data2)

# predict
interpreter1.invoke()
interpreter2.invoke()

# get the result
output_data1 = interpreter1.get_tensor(output_details1[0]['index'])
output_data2 = interpreter2.get_tensor(output_details2[0]['index'])

print('---output details---')
print(output_details1)
print(output_details2)

print('---output data---')
print(output_data1)
print(output_data2)