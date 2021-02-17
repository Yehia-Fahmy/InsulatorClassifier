import numpy as np
import os
import cv2
import time as t
import random
import pickle
import matplotlib.pyplot as plt


# function definitions
# quick function to show the image
def show(img):
    plt.imshow(img, cmap='gray')
    plt.show()


# function to convert the time into something readable
def convert_time(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%d:%02d:%02d" % (hour, minutes, seconds)


# loads the pictures from the local machine
def load_data():
    print('Loading data...')
    err = 0
    for catagory in CATAGORIES:  # for each catagory
        counter = 0  # counter to add every 10th element to testing
        path = os.path.join(IMAGE_PATH, catagory)
        classification = CATAGORIES.index(catagory)
        for img in os.listdir(path):  # for each image
            counter = counter + 1  # index the counter
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # read the image
                img_array = cv2.resize(img_array, (ORIGINAL_SIZE, ORIGINAL_SIZE))  # assert the correct size
                if counter % 10 == 0:
                    TESTING_DATA.append([img_array, classification])  # adds the data to the testing list
                else:
                    TRAINING_DATA.append([img_array, classification])  # adds the data to the training list
            except Exception as e:
                err = err + 1  # counts the errors
    print(f'{len(TRAINING_DATA)} Training images')
    print(f'{len(TESTING_DATA)} Testing images')


# global variables
CATAGORIES = ["HC-1", "HC-2", "HC-3", "HC-4", "HC-5", "HC-6", "HC-7"]
TRAINING_DATA = []
TESTING_DATA = []
IMAGE_PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\Enhanced_Pictures'
ORIGINAL_SIZE = 480


# code to run
start_time = t.time()
print("Starting...")

load_data()

# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# final message
print(f"Finished in: {total_time}")
print('Success!')