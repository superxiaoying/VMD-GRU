# -*- coding:utf8 -*-

import os
import numpy as np
import keras
from keras.utils import custom_object_scope

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# 加载之前的预训练模型，用了nn_utils中__init__.py中的_get_scope_dict()
def load_mask_model(file_name):
    from nn_utils import _get_scope_dict
    with custom_object_scope(_get_scope_dict()):
        model = keras.models.load_model(file_name)
    return model

# plot
def save_fig(fig_path, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(fig_path + "." + fig_extension)
    # print("Saving figure", path)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_files, batch_size=32, n_samples=100, dim=50, n_channels=1,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.n_samples = n_samples # sample numbers in each file
        self.list_files = list_files
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_files) / self.batch_size)) # number of files per epoch

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if (index+1)*self.batch_size < len(self.list_files):
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        else:
            indexes = self.indexes[index*self.batch_size:]
        # Find list of IDs
        list_files_temp = [self.list_files[k] for k in indexes]

        # Generate data
        X, Y = self.__data_generation(list_files_temp)

        return X, Y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_files))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            # print(self.indexes)

    def __scale_normalize(self, data):
        scale = 2/140  # (1 - -1)/(30 - -80) -> 120 - -120
        data_scaled = scale * (data + 80) -1  # scale to (-1 - 1)
        return data_scaled

    def __data_generation(self, list_files_temp):
        'Generates data containing batch_size samples' # X : (batch_size*n_samples, dim, n_channels)
        # Initialization
        batch_n = len(list_files_temp)*self.n_samples
        X_noise = np.empty((batch_n, self.dim, self.n_channels)) #(32*100, 50, 1)
        X_mask = np.empty((batch_n, self.dim, self.n_channels))
        
        Y_clean = np.empty((batch_n, self.dim, 1))
        Y_clean_trend = np.empty((batch_n, self.dim, 1))
        Y_label = np.empty((batch_n, self.dim))  # , dtype=int
        Y_season = np.empty((batch_n, self.dim, 1))  # seasonal_level + deform value
        X_timestamp = np.empty((batch_n, self.dim, 1))
        X_timestamp[:, ] = np.expand_dims(np.arange(self.dim), -1)  # np.arange(self.dim) shape (self.dim,)

        # Generate data
        for i_file, filename in enumerate(list_files_temp):
            # Store sample
            idx = i_file * self.n_samples
            container = np.load(filename)
            data_noise = container['noise'].transpose()
            data_noise_scaled = self.__scale_normalize(data_noise)
            X_noise[idx:idx+self.n_samples, :, 0] = data_noise_scaled
            if self.n_channels > 1:
                seasonal_value = container['seasonal_value'].transpose()
                X_noise[idx:idx+self.n_samples, :, 1] = seasonal_value # [-1, 1]
            data_clean = container['clean'].transpose()
            Y_clean[idx:idx+self.n_samples,] = np.expand_dims(data_clean, -1) # may broadcasting
            data_clean_smooth = container['clean_smooth'].transpose()
            Y_clean_trend[idx:idx+self.n_samples,] = np.expand_dims(data_clean_smooth, -1) # may broadcasting

            data_mask = container['mask'].transpose() 
            X_mask[idx:idx+self.n_samples, :, 0] = data_mask 
            if self.n_channels > 1:
                X_mask[idx:idx+self.n_samples, :, 1] = 1 # may broadcasting to multi-dimension, all 1 for seasonal signal
            # Store class
            data_label = container['label'].transpose() # change from 1000 to 10, after u_law projection
            Y_label[idx:idx+self.n_samples,] = data_label

            seasonal_value = container['seasonal_value'].transpose()
            seasonal_level = container['seasonal_level'].transpose()
            Y_season[idx:idx+self.n_samples, ] = np.expand_dims(seasonal_value*seasonal_level, axis=-1)
        
        Y_oh = keras.utils.to_categorical(Y_label, num_classes=self.n_classes)

        X = [X_noise, X_mask, X_timestamp] #, X_mask, X_timestamp
        Y = [Y_clean_trend, Y_season, Y_clean, Y_oh]  # , Y_dv, Y_clean_smooth, y_season, y_deform] 

        return X, Y
