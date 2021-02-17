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
                TRAINING_DATA.append([img_array, class_num])  # adds the data as a list
            except Exception as e:
                err = err + 1  # counts the errors we have
        print(len(TRAINING_DATA), "training examples (", err, "errors )")


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
    size = 0
    for features, label in data:  # splits the data
        size = features.shape[1]
        X.append(features)
        y.append(label)
    X = np.array(X).reshape(-1, size, size, 3)
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
    print(f'{len(new_data)} images')
    return new_data


# a function to crop the image into smaller images
def crop_data(data):
    print("Cropping data...")
    err = 0             # variable to keep track of any missed images
    new_data = []       # list to hold the data
    try:
        for img in data:                # for each image
            for i in range(round(NEW_SQUARE_DIM / IMG_SIZE)):          # going through the rows
                for k in range(round(NEW_SQUARE_DIM / IMG_SIZE)):      # going through the columns
                    new_data.append([img[0][i * IMG_SIZE:(i + 1) * IMG_SIZE, k * IMG_SIZE:(k + 1) * IMG_SIZE],
                                     img[1]])  # adds the data as a list
    except Exception as e:
        err = err + 1
    print(f'Finished cropping with {err} errors')
    print(f'{len(new_data)} images')
    return new_data


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
    new_data = crop_data(data)
    new_data = rotate_data(new_data)
    new_data = flip_data(new_data)
    return new_data


# global variables
CATEGORIES = ['Class (1)', 'Class (2)', 'Class (3)', 'Class (4)', 'Class (5)', 'Class (6)', 'Class (7)']

TRAINING_DATA = []
TESTING_DATA = []
ORIGINAL_HEIGHT = 3216
ORIGINAL_WIDTH = 4228
NEW_SQUARE_DIM = 244 * 9
IMG_SIZE = 224
# path to training photos
TRAINING_PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\DatabaseOrganized'
# path to testing photos
TESTING_PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\Testing_DatabaseOrganized'

# code to run
start_time = t.time()
print("Starting...")

load_data()
load_testing_data()

TRAINING_DATA = crop_middle(TRAINING_DATA)
TESTING_DATA = crop_middle(TESTING_DATA)

TRAINING_DATA = increase_data(TRAINING_DATA)
TESTING_DATA = increase_data(TESTING_DATA)

TRAINING_DATA = shuffle_data(TRAINING_DATA)
TESTING_DATA = shuffle_data(TESTING_DATA)

images, labels = split_data(TRAINING_DATA)
testing_images, testing_labels = split_data(TESTING_DATA)

save_data(images, labels)
save_testing_data(testing_images, testing_labels)

# prints the elapsed time for convenience
total_time = t.time() - start_time
total_time = round(total_time, 2)
total_time = convert_time(total_time)

# final message
print(f"Finished in: {total_time}")
print('Success!')
