import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

import keras
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import Input, Dense, Flatten, Dropout, Reshape
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.layers.recurrent import GRU
from keras.callbacks import ModelCheckpoint
from sklearn.utils import shuffle

import os
import time

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Convolutional Neural Net Training Script (for Lindy, Balboa, Shag, Charleston)
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''

def scale_range(arrays, min_, max_):
    '''
        Standardizes input data between two user specified values
        INPUT:
            -  arrays : array(s)
            -  min_   : minumum scale value (int)
            -  max_   : maximum scale value (int)
        OUTPUT:
            - arrays  : re-scaled array(s)

    '''
    arrays += -(np.min(arrays))
    arrays /= np.max(arrays) / (max_ - min_)
    arrays += min_
    return arrays

def preprocess_data(X,Y,reduce_to=0):
    '''
        Imports data, slices and binerazes and checks X and Y shapes

        INPUT:
            -  X      : npz file mel-spectrograms
            -  Y      : npz file labels
            -  reduce_to : number of samples to train with (int) default len(X)

        OUTPUT:
            -  X      : array mel-spectrograms (timmed if applicable)
            -  Y      : array mel-spectrograms (timmed, transformed
                                                   if applicable)

    '''
    # Import Data
    X = np.load(X)['arr_0']
    Y = np.load(Y)['arr_0']

    if reduce_to != 0:
        # Reduce to data points
        X = X[:reduce_to, :, :, :]
        Y = Y[:reduce_to]
    # Checks labels
    print "Y data: {}".format(Y[:10])


    binarize = str(raw_input("Do you need to binarize categories? (y/n):"))
    if binarize == 'y':
        # Convert Y's to binary categories
        encoder = LabelBinarizer()
        Y = encoder.fit_transform(Y)

    # Normalize data between -1 and 1
    X = scale_range(X,-1,1)

    print "X data: {}".format(X[0, :, :, :])

    # Check input and label shape
    print X.shape, Y.shape

    return X, Y


def test_train_split(X,Y,proportion=0.8):
    '''
        Splits dataset to training and testing arrays

        INPUT:
            -  X      : mel-spectrogram (array)
            -  Y      : prepared labels (array)
            -  proportion : proportion of dataset for testing (float)

        OUTPUT:
            -  X_train, y_train  : normalized training data and labels (array)
            -  X_test, y_test    : normalized testing data and labels (array)
            -  input_shape       : tuple of dimensions for X

    '''
    # Get index of split
    split = int(round(len(X)*proportion,0))

    # Test Train Split
    X_train = X[:split]
    X_test = X[split:]

    X_train = shuffle(X_train,random_state=8)
    X_test = shuffle(X_test,random_state=14)

    y_train = Y[:split]
    y_test = Y[split:]

    y_train = shuffle(y_train,random_state=8)
    y_test = shuffle(y_test,random_state=14)

    print "y_test (shuffled): {}".format(y_test[0:10])
    print '''X_train shape: {}\n y_train shape: {}\n X_test shape:
             {}\n y_test shape: {}'''.format(X_train.shape,
                                             y_train.shape,
                                             X_test.shape,
                                             y_test.shape)

    # Define Input Shape of each sample
    input_shape = (X.shape[1], X.shape[2], X.shape[3])

    return X_train, y_train, X_test, y_test, input_shape


def Model(num_classes, input_shape):

    ''' Convolutional Neural Network Model

        - Zero padding
        - 4 hidden layers
        - ELU activation Function (alpha 1.4)
        - Adam Optimization (learing rate 0.01)
        - Gated Recurrent Unit

        '''

    model = Sequential()

    # Axis
    freq_axis = 1
    sample_axis = 3

    # Input layer
    model.add(ZeroPadding2D(input_shape=input_shape,padding=(0, 37)))
    model.add(BatchNormalization(axis=freq_axis))

    # Hidden Layer 1
    model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1),
                                             input_shape=input_shape,
                                             border_mode='same',
                                             bias_initializer=Constant(0.01)))
    model.add(BatchNormalization(axis=sample_axis))
    model.add(ELU(alpha=1.4))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(0.1))

    # Hidden Layer 2
    model.add(Conv2D(128, (3, 3), border_mode='same',
                                  bias_initializer=Constant(0.01)))
    model.add(BatchNormalization(axis=sample_axis))
    model.add(ELU(alpha=1.4))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(0.1))

    # Hidden Layer 3
    model.add(Conv2D(128, (3, 3), border_mode='same',
                                  bias_initializer=Constant(0.01)))
    model.add(BatchNormalization(axis=sample_axis))
    model.add(ELU(alpha=1.4))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(3, 3)))
    model.add(Dropout(0.1))

    # Hidden Layer 4
    model.add(Conv2D(128, (3, 3), border_mode='same',
                                  bias_initializer=Constant(0.01)))
    model.add(BatchNormalization(axis=sample_axis))
    model.add(ELU(alpha=1.4))
    model.add(MaxPooling2D(pool_size=(1, 1), strides=(3, 3)))
    model.add(Dropout(0.1))

    # Reshape for GRU
    model.add(Flatten())
    model.add(Reshape((34, 128)))

    # Hidden GRU layer
    model.add(GRU(64, return_sequences=True))
    model.add(GRU(64, return_sequences=False))
    model.add(Dropout(0.3))

    # Output layer
    model.add(Dense(num_classes, activation='softmax'))

    binary = int(raw_input("how many columns for y-matrix? (e.g. 6):"))
    if binary > 1:
        loss_type = keras.losses.categorical_crossentropy
    else:
        loss_type = keras.losses.mean_squared_error

    model.compile(loss=loss_type,
                  optimizer=keras.optimizers.Adam(lr=0.01),
                  metrics=['accuracy'])
    return model


def save_model_and_metrics(model, run_time_s):
    '''
        Saves model setup in json format and weights in h5 format.

        INPUT:
            -  model      : trained CRNN model
            -  run_time_s : run time of model (seconds)

        OUTPUT:
            None

    '''
    # Save model to JSON
    model_json = model.to_json()
    filename = "CNN_Models/model-{}".format(str(
                raw_input("Enter Model ID in numeric format (e.g. 001): ")))

    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)

    # Save weights to HDF5
    model.save_weights(filename + ".h5")
    print("Saved model to disk \n")

    # Save Params
    with open(filename + "-params.txt", "w") as param_file:
          param_file.write("Run time (seconds): {}".format(run_time_s))

    summary_table = model.summary()

    with open(filename + "-params.txt", "w") as param_file:
          param_file.write(str(summary_table))



if __name__ == '__main__':
    os.system("mkdir CNN_Models")
    X, Y = preprocess_data(raw_input("Enter url for X.npy :"),
                           raw_input("Enter url for Y.npy:"),
                           int(raw_input('''Limit samples to n? if yes enter
                                            number (int), if no enter 0:''')))

    X_train, y_train, X_test, y_test, input_shape = test_train_split(X, Y,
                    float(raw_input("Enter train/test proportion (e.g. 0.8):")))

    batch_size = int(raw_input("Enter batch size (int) :"))
    num_classes = y_train.shape[1]
    epochs = int(raw_input("Enter number of epochs (int) :"))

    model = Model(num_classes, input_shape)

    # Record model start time
    start_time = time.clock()

     # Initialize weights using checkpoint if it exists. (Checkpointing requires h5py)
    # load_checkpoint = bool(raw_input("do you want to load checkpoint weights? True/False"))
    # if load_checkpoint:
    #     model.load_weights('weights.hdf5')
    # else:
    #     continue

    checkpointer = ModelCheckpoint(filepath='weights.hdf5', verbose=1,
                                   save_best_only=True)

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[checkpointer],
              shuffle=False)

    score = model.evaluate(X_test, y_test, verbose=0)
    model.summary()

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Record model stop time
    run_time_s = time.clock() - start_time

    print run_time_s
    print "seconds"

    # print "Close chart to continue"
    # plt.plot(range(1, epochs+1), history.acc)
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.show()

    save_model_and_metrics(model, run_time_s)
