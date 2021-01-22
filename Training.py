import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model as M
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
import cv2
import os
from matplotlib import pyplot as plt
import time as t
import pickle
from keras.utils import to_categorical


# global variables
CATAGORIES = ['Class (1)', 'Class (2)', 'Class (3)', 'Class (4)', 'Class (5)', 'Class (6)', 'Class (7)']
DATA = []
TESTING_DATA = []
IMG_SIZE = 224
# path to training photos
TRAINING_PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\DatabaseOrganized'
# path to testing photos
TESTING_PATH = r'C:\Users\Yehia\OneDrive - University of Waterloo\Winter 2021 Co-op\Testing_DatabaseOrganized'