import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import cv2
import os

tf.config.set_visible_devices([], 'GPU')
# Set the TensorFlow logging level to suppress warnings
tf.get_logger().setLevel('ERROR')  # or 'FATAL' for even fewer messages

#Train and test:

#For train
train_ds = keras.utils.image_dataset_from_directory('brain-mri/Training')

#For validation
val_ds = keras.utils.image_dataset_from_directory('brain-mri/Testing',)

#Convert to numpy iterator
data_iterator_ts = train_ds.as_numpy_iterator()
data_iterator_vs = val_ds.as_numpy_iterator()
