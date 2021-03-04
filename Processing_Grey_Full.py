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


# rotates the images in all orientations
def rotate_data(data):
    print("Rotating data...")
    new_data = []  # list to hold the data
    err = 0  # variable to keep track of any missed images
    try:
        for img in data:
            new_data.append(img)  # adds the original
            new_data.append([cv2.rotate(img[0], cv2.ROTATE_90_CLOCKWISE), img[1]])
            new_data.append([cv2.rotate(img[0], cv2.ROTATE_90_COUNTERCLOCKWISE), img[1]])
            new_data.append([cv2.rotate(img[0], cv2.ROTATE_180), img[1]])  # adds the other 3 orientations
    except Exception as e:
        err = err + 1
    print(f'{len(new_data)} images')
    return new_data


# flips the images in all orientations
def flip_data(data):
    print("Flipping data...")
    new_data = []       # list to hold the data
    err = 0             # variable to keep track of any missed images
    try:
        for img in data:
            new_data.append([img[0], img[1]])  # adds the original image
            new_data.append([np.flip(img[0], axis=0), img[1]])  # adds the image flipped horizontally
            new_data.append([np.flip(img[0], axis=1), img[1]])  # adds the image flipped vertically
            new_data.append([np.flip(np.flip(img[0], axis=1), axis=0), img[1]])  # adds the image flipped both ways
    except Exception as e:
        err = err + 1
    print(f'{len(new_data)} images')
    return new_data


# runs all the functions that can increase the numbers of training data
def increase_data(data):
    print('Increasing data...')
    new_data = rotate_data(data)
    new_data = flip_data(new_data)
    return new_data


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
    size = 0
    for features, label in data:  # splits the data
        size = features.shape[1]
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, ORIGINAL_SIZE, ORIGINAL_SIZE, 1)
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


# global variables
CATAGORIES = ["HC-1", "HC-2", "HC-3", "HC-4", "HC-5", "HC-6", "HC-7"]
TRAINING_DATA = []
TESTING_DATA = []
IMAGE_PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\Enhanced_Pictures'
ORIGINAL_SIZE = 481
IMG_SIZE = 224


# code to run
start_time = t.time()
print("Starting...")

load_data()

print('---Training Data---')
TRAINING_DATA = increase_data(TRAINING_DATA)
TRAINING_DATA = shuffle_data(TRAINING_DATA)
images, labels = split_data(TRAINING_DATA)
save_data(images, labels)

print('---Testing Data---')
TESTING_DATA = increase_data(TESTING_DATA)
TESTING_DATA = shuffle_data(TESTING_DATA)
testing_images, testing_labels = split_data(TESTING_DATA)
save_testing_data(testing_images, testing_labels)


# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# final message
print(f"Finished in: {total_time}")
print('Success!')