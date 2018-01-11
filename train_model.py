
import pandas as pd
import numpy as np
import requests
import random
import os
from boto import connect_s3


from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.preprocessing import LabelBinarizer


def get_npy_from_s3(bucket,file_path):

    # Import bash_profile keys
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

    conn =  connect_s3(access_key,secret_key)
    bucket = conn.get_bucket(bucket)
    content = bucket.get_key(file_path).get_contents_as_string()

    return np.load(StringIO(content))

def test_train_split(X,Y, split=0.8):

    split = int(round(len(Y)*0.8,0))

    X_train = X[:split]
    X_test = X[split:]

    Y_train = Y[:split]
    Y_test = Y[split:]

    # Normalize inputs from 0-255 to 0-1
    X_train = np.expand_dims(X_train / 255, axis=1)
    X_test = np.expand_dims(X_test / 255, axis=1)

    #Convert Y's to binary categories
    encoder = LabelBinarizer()
    Y_train = encoder.fit_transform(Y_train)
    Y_test = encoder.fit_transform(Y_test)

    return X_train, Y_train, X_test, Y_test

def fit_model(X_train, Y_train, X_test, Y_test):

    model = Sequential()
    # Dense(600) is a fully-connected layer with 600 hidden units.
    # in the first layer, you must specify the expected input data shape (no. of features):

    #Hidden Layer 1
    model.add(Dense(600, activation='sigmoid',input_shape=2))
    model.add(Dropout(0.3))
    #Hidden Layer 2
    model.add(Dense(300, activation='sigmoid'))
    model.add(Dropout(0.3))
    #Hidden Layer 3
    model.add(Dense(300, activation='sigmoid'))
    model.add(Dropout(0.1))
    #Hidden Layer 4
    model.add(Dense(6, activation='softmax'))

    sgd = SGD(lr=0.6, decay=1e-5, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Fit the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=50, verbose=1)

    score = model.evaluate(X_test, Y_test)

    return model


if __name__ == '__main__':
    bucket, X_data, Y_data = "swingmusic001", "metadata/processed_X_data.csv","metadata/processed_Y_data.csv"
    X = get_npy_from_s3(bucket,X_data)
    Y = get_npy_from_s3(bucket,Y_data)
    X_train, Y_train, X_test, Y_test = test_train_split(X, Y)
    print Y_test
    print "ready"
    #fit_model(X_train, Y_train, X_test, Y_test)
