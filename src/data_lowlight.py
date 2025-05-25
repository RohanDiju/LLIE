import numpy as np
import tensorflow as tf
from tensorflow import keras
import random
import glob
from PIL import Image

def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.JPG")
    train_list = image_list_lowlight
    random.shuffle(train_list)
    return train_list

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, lowlight_images_path, batch_size):
        self.train_list = populate_train_list(lowlight_images_path)
        self.size = 512
        self.data_list = self.train_list
        self.batch_size = batch_size
        self.n_channels = 3
        print("Total training examples:", len(self.train_list))    

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.train_list) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.data_list[index * self.batch_size:(index + 1) * self.batch_size]

        # Generate data
        X = self.__data_generation(indexes)

        return X

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.data_list))
        if getattr(self, 'shuffle', False):  # Make sure shuffle is defined
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' 
        # Initialization
        X = np.empty((self.batch_size, self.size, self.size, self.n_channels))

        # Generate data
        for i, data_lowlight_path in enumerate(indexes):
            # Load and preprocess image
            data_lowlight = Image.open(data_lowlight_path)
            data_lowlight = data_lowlight.resize((self.size, self.size), Image.LANCZOS)
            data_lowlight = np.asarray(data_lowlight) / 255.0
            X[i] = data_lowlight

        return tf.convert_to_tensor(X)
