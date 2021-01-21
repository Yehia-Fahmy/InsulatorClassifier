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
    path = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\Code\First_App'  # path to directory with the all pictures
    for catagory in CATAGORIES:  # for every catagory
        folder = os.path.join(path, catagory)   # joins folder with images
        class_num = CATAGORIES.index(catagory)  # 0 for cat 1 for dog
        counter = 0
        for img in os.listdir(folder):  # for every image
            if counter < 1000:
                try:
                    img_array = cv2.imread(os.path.join(folder, img), cv2.IMREAD_GRAYSCALE)  # reads the image
                    img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # confirms it is the correct size
                    DATA.append([img_array, class_num])  # adds the data as a list
                    counter = counter + 1
                except Exception as e:
                    err = err + 1  # counts the errors we have
            else:
                break
        print(len(DATA), "training examples (", err, "errors )")


# load testing data
def load_testing_data():
    print("Loading testing data...")
    err = 0  # variable to keep track of any missed images
    path = PATH
    for catagory in TESTING_CATAGORIES:  # for every catagory
        folder = os.path.join(path, catagory)   # joins folder with images
        class_num = TESTING_CATAGORIES.index(catagory)  # 0 for cat 1 for dog
        for img in os.listdir(folder):  # for every image
            try:
                img_array = cv2.imread(os.path.join(folder, img), cv2.IMREAD_GRAYSCALE)  # reads the image
                img_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # confirms it is the correct size
                TESTING_DATA.append([img_array, class_num])  # adds the data as a list
            except Exception as e:
                err = err + 1  # counts the errors we have
        print(len(TESTING_DATA), "testing examples (", err, "errors )")



# shuffles the data
def shuffle_data(data):
    print("Shuffling data...")
    random.shuffle(data)        # randomly shuffles the data
    return data


# splits into lables and features
def split_data(data):
    print("Splitting data...")
    X = []      # list of images
    y = []      # list of labels
    for features, label in data:        # splits the data
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE)
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


# global variables
CATAGORIES = ['cats', 'dogs']
TESTING_CATAGORIES = ['test_cats', 'test_dogs']
DATA = []
TESTING_DATA = []
IMG_SIZE = 224
PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\DatabaseOrganized'  # path to directory with the all pictures

