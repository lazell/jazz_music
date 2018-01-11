import numpy as np
import pandas as pd

import keras
from keras.datasets import mnist
from keras.layers import Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pylab as plt

def preprocess_data(X,Y,reduce_to=False):
    #Import Data
    X = np.load('/Users/katielazell-fairman/desktop/projects/jazz_music/micro_subset/data/processed_X_data3.npy')
    Y = np.load('/Users/katielazell-fairman/desktop/projects/jazz_music/micro_subset/data/processed_Y_data3.npy')

    if reduce_to not False:
        #Reduce to data points
        X = X[:reduce_to, :, :, :]
        Y = Y[:reduce_to]
    else:
        continue

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
    return X_train.shape, y_train.shape, X_test.shape, y_test.shape

    #Define Input Shape of each sample
    input_shape = (128, 1292, 1)



def Model(num_classes):
    num_classes = 6

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

def fit(model, X_train,y_train, batch_size, epochs):
    pass




if __name__ == '__main__':
    preprocess_data(raw_input("Enter url for X.npy :"), raw_input("Enter url for X.npy:"))
    X_train, y_train, X_test, y_test = test_train_split(X,Y,float(raw_input("Enter test/train proportion (e.g. 0.8):"))

    batch_size = int(raw_input("Enter batch size (int) :")
    num_classes = int(raw_input("Enter number of classes (int) :")
    epochs = int(raw_input("Enter number of epochs (int) :")

    model = Model(num_classes)

    model.fit(X_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=[history],
              shuffle=True)
