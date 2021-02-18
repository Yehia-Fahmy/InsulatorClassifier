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