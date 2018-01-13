import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pylab as plt

import os
import time

def preprocess_data(X,Y,reduce_to=0):
    #Import Data
    X = np.load(X)['arr_0']
    Y = np.load(Y)['arr_0']

    if reduce_to != 0:
        #Reduce to data points
        X = X[:reduce_to, :, :, :]
        Y = Y[:reduce_to]

    #Remove NaNs
    Y = pd.DataFrame(Y)
    Y = Y.fillna('None')
    Y = np.array(Y[0])

    print "Y data: {}".format(Y[:10])
    print "X data: {}".format(X[0, :, :, :])

    binarize = str(raw_input("Do you need to binarize categories? (y/n):"))
    if binarize == 'y':

        #Convert Y's to binary categories
        encoder = LabelBinarizer()
        Y = encoder.fit_transform(Y)
    else:
        #Transform
        Y = np.reshape(Y,(len(Y),1))

    print X.shape, Y.shape
    return X, Y


def test_train_split(X,Y,proportion=0.8):
    # Get index of split
    split = int(round(len(X)*proportion,0))

    # Test Train Split
    X_train = X[:split]
    X_test = X[split:]

    y_train = Y[:split]
    y_test = Y[split:]

    print "X_train shape: {}\n y_train shape: {}\n X_test shape: {}\n y_test shape: {}".format(X_train.shape,y_train.shape, X_test.shape, y_test.shape)

    #Define Input Shape of each sample
    input_shape = (X.shape[1], X.shape[2], X.shape[3])

    return X_train, y_train, X_test, y_test, input_shape


def Model(num_classes, input_shape):


    #Generate Model
    model = Sequential()

    #Input layer
    model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1),
                     activation='relu',
                     input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 2)))

    #Hidden Layer 1
    model.add(Conv2D(384, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 5)))

    #Hidden Layer 2
    model.add(Conv2D(768, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 8)))

    #Hidden Layer 3
    model.add(Conv2D(1536, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 8)))

    #Hidden Layer 4
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid'))

    #Output layer
    model.add(Dense(num_classes, activation='softmax'))

    binary = int(raw_input("how many columns for y-matrix? (e.g. 6):"))
    if binary < 1:
        loss_type = keras.losses.categorical_crossentropy
    else:
        loss_type = keras.losses.mean_squared_error

    model.compile(loss=loss_type,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def fit(model, X_train,y_train, batch_size, epochs):
    pass

def save_model_and_metrics(model, history, run_time_s):
    # Save model to JSON
    model_json = model.to_json()
    filename = "CNN_Models/model-{}".format(str(raw_input("Enter Model ID in numeric format (e.g. 001): ")))
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)

    # Save weights to HDF5
    model.save_weights(filename + ".h5")
    print("Saved model to disk \n")

    #Save Params
    with open(filename + "-params.txt", "w") as param_file:
          param_file.write(str(history.params))
          param_file.write("Run time (seconds): {}".format(run_time_s))


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


if __name__ == '__main__':
    os.system("mkdir CNN_Models")
    X, Y = preprocess_data(raw_input("Enter url for X.npy :"),
                           raw_input("Enter url for Y.npy:"),
                           int(raw_input("Limit samples to n? if yes enter number (int), if no enter 0:")))

    X_train, y_train, X_test, y_test, input_shape = test_train_split(X,Y,float(raw_input("Enter train/test proportion (e.g. 0.8):")))

    batch_size = int(raw_input("Enter batch size (int) :"))
    num_classes = y_train.shape[1]
    epochs = int(raw_input("Enter number of epochs (int) :"))

    model = Model(num_classes, input_shape)
    history = AccuracyHistory()

    # Record model start time
    start_time = time.clock()

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[history],
              shuffle=True)

    score = model.evaluate(X_test, y_test, verbose=0)

    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Record model stop time
    run_time_s = time.clock() - start_time
    if run_time_s >=9000:
        print 1.*run_time_s/3600, " Hours to run model"
    else:
        print 1.*run_time_s/60, " Minutes to run model"

    print "Close chart to continue"
    plt.plot(range(1, epochs+1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    save_model_and_metrics(model, history, run_time_s)
