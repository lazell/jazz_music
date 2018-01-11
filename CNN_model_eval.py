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
    X = np.load(X)
    Y = np.load(Y)

    if reduce_to != 0:
        #Reduce to data points
        X = X[:reduce_to, :, :, :]
        Y = Y[:reduce_to]

    #Remove NaNs
    Y = pd.DataFrame(Y)
    Y = Y.fillna('None')
    Y = np.array(Y[0])

    #Convert Y's to binary categories
    encoder = LabelBinarizer()
    Y = encoder.fit_transform(Y)

    return X, Y

def test_train_split(X,Y,proportion=0.8):
    # Get index of split
    split = int(round(len(X)*proportion,0))

    # Test Train Split
    X_train = X[:split]
    X_test = X[split:]

    y_train = Y[:split]
    y_test = Y[split:]

    print "X_train shape: {}\n y_train shape: {}\n X_test shape: {}\n y_test shape: {}\n".format(X_train.shape,
                                                                                                 y_train.shape,
                                                                                                 X_test.shape,
                                                                                                 y_test.shape)
    return X_train, y_train, X_test, y_test

    #Define Input Shape of each sample
    input_shape = (128, 1292, 1)



def Model(num_classes):


    #Define Input Shape of each sample
    input_shape = (128, 1292, 1)

    #Generate Model
    model = Sequential()

    #Input Layer
    model.add(Conv2D(120, kernel_size=(3, 3), strides=(1, 1),
                     activation='softmax',
                     input_shape=input_shape))

    model.add(MaxPooling2D(pool_size=(2, 4), strides=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='softmax'))
    model.add(MaxPooling2D(pool_size=(2, 4)))

    model.add(Flatten())
    model.add(Dense(1000, activation='softmax'))

    #Output layer
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model

def fit(model, X_train,y_train, batch_size, epochs):
    pass

def save_model_and_metrics(model, history, run_time_s):
    # Save model to JSON
    model_json = model.to_json()
    filename = "CNN_Models/model-{}".format(str(raw_input("Enter Model ID in numeric format (e.g. 001)")))
    with open(filename + ".json", "w") as json_file:
        json_file.write(model_json)

    # Save weights to HDF5
    model.save_weights(filename + ".h5")
    print("Saved model to disk")

    #Save Params
    with open(filename + "-params.txt", "w") as param_file:
          param_file.write(history.params)
          param_file.write("Run time (seconds): {}".format(run_time_s)


class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.acc = []

    def on_epoch_end(self, batch, logs={}):
        self.acc.append(logs.get('acc'))


if __name__ == '__main__':
    os.system("mkdir CNN_Models")
    X, Y = preprocess_data(raw_input("Enter url for X.npy :"),
                           raw_input("Enter url for Y.npy:"),
                           int(raw_input("Limit samples to n? if yes enter number (int), if no enter 0:\n)))

    X_train, y_train, X_test, y_test = test_train_split(X,Y,float(raw_input("Enter train/test proportion (e.g. 0.8):")))

    batch_size = int(raw_input("Enter batch size (int) :"))
    num_classes = y_train.shape[1]
    epochs = int(raw_input("Enter number of epochs (int) :"))

    model = Model(num_classes)
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


    plt.plot(range(1, epochs+1), history.acc)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()
    print "Close chart to continue"

    save = raw_input("Would you like to save this model & it's metrics? (y/n): ")
    if save == 'y':
        save_model_and_metrics(model, history, run_time_s)
