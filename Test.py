# this file will attempt to scan the image in a higher resolution

import numpy as np
import os
import cv2
import time as t
import random
import pickle
import matplotlib.pyplot as plt


# function definitions
# loads the pictures from the local machine
def load_data():
    print("Loading data...")
    err = 0  # variable to keep track of any missed images
    path = TRAINING_PATH
    for category in CATEGORIES:  # for every category
        folder = os.path.join(path, category)  # joins folder with images
        class_num = CATEGORIES.index(category)  # gives each folder a class number
        for img in os.listdir(folder):  # for every image
            try:
                img_array = cv2.imread(os.path.join(folder, img), cv2.IMREAD_COLOR)  # reads the image
                img_array = cv2.resize(img_array, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))  # confirms it is the correct size
                DATA.append([img_array, class_num])  # adds the data as a list
            except Exception as e:
                err = err + 1  # counts the errors we have
        print(len(DATA), "training examples (", err, "errors )")


# load testing data
def load_testing_data():
    print("Loading testing data...")
    err = 0  # variable to keep track of any missed images
    path = TESTING_PATH
    for category in CATEGORIES:  # for every category
        folder = os.path.join(path, category)  # joins folder with images
        class_num = CATEGORIES.index(category)
        for img in os.listdir(folder):  # for every image
            try:
                img_array = cv2.imread(os.path.join(folder, img), cv2.IMREAD_COLOR)  # reads the image
                img_array = cv2.resize(img_array, (ORIGINAL_WIDTH, ORIGINAL_HEIGHT))  # confirms it is the correct size
                TESTING_DATA.append([img_array, class_num])  # adds the data as a list
            except Exception as e:
                err = err + 1  # counts the errors we have
        print(len(TESTING_DATA), "testing examples (", err, "errors )")


# shuffles the data
def shuffle_data(data):
    print("Shuffling data...")
    random.shuffle(data)  # randomly shuffles the data
    return data


# splits into lables and features
def split_data(data):
    print("Splitting data...")
    X = []  # list of images
    y = []  # list of labels
    for features, label in data:  # splits the data
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
    return X, y


# dumps the pictures
def save_data(X, y):
    print("Saving...")
    pickle_out = open("Images.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Labels.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


# dumps the testing pictures
def save_testing_data(X, y):
    print("Saving...")
    pickle_out = open("Testing_Images.pickle", "wb")
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open("Testing_Labels.pickle", "wb")
    pickle.dump(y, pickle_out)
    pickle_out.close()


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


# function to take the middle portion of each image
def crop_middle(data):
    print("Cropping data...")
    new_data = []
    err = 0
    try:
        for pair in data:
            img = pair[0]
            label = pair[1]
            y, x, z = img.shape
            startx = int(x / 2 - (NEW_SQUARE_DIM / 2))
            starty = int(y / 2 - (NEW_SQUARE_DIM / 2))
            new_img = img[starty:starty+NEW_SQUARE_DIM, startx:startx+NEW_SQUARE_DIM, 0:3]
            new_data.append([new_img, label])
    except Exception as e:
        err += 1

    print(f"Finished cropping with {err} errors")
    return new_data


# global variables
CATEGORIES = ['Class (1)']
''', 'Class (2)', 'Class (3)', 'Class (4)', 'Class (5)', 'Class (6)', 'Class (7)'''

DATA = []
TESTING_DATA = []
ORIGINAL_HEIGHT = 3216
ORIGINAL_WIDTH = 4228
NEW_SQUARE_DIM = 244 * 10
IMG_SIZE = 224
# path to training photos
TRAINING_PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\DatabaseOrganized'
# path to testing photos
TESTING_PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\Testing_DatabaseOrganized'

# code to run
start_time = t.time()
print("Starting...")

# load_data()
load_testing_data()

TESTING_DATA = crop_middle(TESTING_DATA)

'''DATA = shuffle_data(DATA)
TESTING_DATA = shuffle_data(TESTING_DATA)'''

'''images, labels = split_data(DATA)
testing_images, testing_labels = split_data(TESTING_DATA)

save_data(images, labels)
save_testing_data(testing_images, testing_labels)'''

# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# final message
print(f"Finished in: {total_time}")
print('Success!')
