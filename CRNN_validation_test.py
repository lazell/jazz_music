import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from keras.models import model_from_json
from sklearn import metrics

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Fetches Convolutional Neural Net model (for Lindy, Balboa, Shag, Charleston)
and validates model with reserved test dataset
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

def load_validation_data():

    ''' Loads and pre-processes validation X and y data

        OUTPUT:
              - X_test   : scaled test mel spectrogram arrays (array)
              - y_test   : test labels (array)
              - df       : test metadats (dataframe)
    '''
    fpath = '/Users/katielazell-fairman/desktop/projects/jazz_music/'

    # Load song metadata dataframe
    df = pd.read_csv(fpath+'processed_data/processed_metadata10.csv')

    # Load processed mel-spectrogram arrays
    arr_test = (np.load(fpath+'processed_data/processed_X_data10.npz')
                        ['arr_0'][1900:2100])
    # Load binerized song labels
    y_test = (np.load('processed_data/processed_Y_sty_data10.npz')
                        ['arr_0'][1900:2100])

    # Use Standardizing array values to fit normalization
    std_arr = (np.load(fpath+'processed_data/standardizing_X_array10.npz')
                        ['arr_0'])
    std = np.expand_dims(std_arr, axis=0)

    # Combine Stardardizing array with mel-spectrograms
    arr_test = arr_test.reshape((200, 128, 1292))
    std_and_arr = np.vstack([std,arr_test])

    # Normalize/Scaled between -1 and 1
    std_and_arr_scaled = scale_range(std_and_arr,-1,1)

    # After scaling remove standardizing Array and re-dimension
    X = std_and_arr_scaled[1:,:,:]
    X_test = X.reshape((200, 128, 1292,1))

    # Shuffle test set
    X_test = shuffle(X_test, random_state=14)
    y_test = shuffle(y_test, random_state=14)


    return X_test, y_test, df


def crnn(X_test):

    '''  Convolutional Neural Net: predicts for Lindy, Balboa, Shag, Charleston


        INPUT:  arr_test     :   Array of pre-normalized mel-spectrograms
                                 (dimensions n x 22) for testing

        OUTPUT: df_pred_crnn :   Dataframe of predictions
                                 (dimensions m x 4) rows in format:
                                 [Lindy, Balboa, Charleston, Shag,display_text]
        '''

    # Create CRNN Model from saved json file
    fpath ='/Users/katielazell-fairman/desktop/projects/jazz_music/CNN_Models/'
    with open(fpath+'model-0123024.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        # load model
        crnn_model = model_from_json(loaded_model_json)

    # LOoad weights into new model
    crnn_model.load_weights(fpath+"model-0123024.h5")
    print("Loaded model with weights")

    # Compile & evaluate loaded model on test data
    crnn_model.compile(loss='categorical_crossentropy',
                       optimizer='adam',
                       metrics=['accuracy'])

    y_pred_crnn = crnn_model.predict(X_test)

    # Save predictuions to DataFrame
    df_pred_crnn = pd.DataFrame(y_pred_crnn)
    df_pred_crnn.columns = ['Lindy', 'Balboa', 'Charleston', 'Shag']

    return df_pred_crnn

if __name__ == '__main__':
    # Load validation data
    X_test, y_test, df = load_validation_data()
    print "loading data to model..."

    # Generate validationpredictions dataframe
    df_pred_crnn = crnn(X_test)

    # Print accuracy score
    y_pred_arr = np.around(np.array(df_pred_crnn),decimals =1)
    accuracy = ((y_test + y_pred_arr) > 1.3).sum()*1./len(y_test)
    print "\n CRNN Model accuracy: {}\n\n".format(round(accuracy,3))

    # Print prediction results (up to first 30) if probability > 25%
    for i, x in enumerate(y_pred_arr[:30]):
        print "\n\nsong: {}\n".format(i)
        if df_pred_crnn.iloc[i][0] > 0.25:
            print "Lindy",round(df_pred_crnn.iloc[i][0],2)*100, "%"

        if df_pred_crnn.iloc[i][1] > 0.25:
            print "Balboa ",round(df_pred_crnn.iloc[i][1],2)*100, "%"

        if df_pred_crnn.iloc[i][2] > 0.25:
            print "Shag", round(df_pred_crnn.iloc[i][2],2)*100, "%"

        if df_pred_crnn.iloc[i][3] > 0.25:
            print "Charleston", round(df_pred_crnn.iloc[i][3],2)*100, "%"
