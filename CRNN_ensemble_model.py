import pandas as pd
import numpy as np

from sklearn.utils import shuffle
from keras.models import model_from_json
from sklearn import metrics



def scale_range(input_, min_, max_):
    input_ += -(np.min(input_))
    input_ /= np.max(input_) / (max_ - min_)
    input_ += min_
    return input_

def load_training_data():
    fpath = '/Users/katielazell-fairman/desktop/projects/jazz_music/'
    df = pd.read_csv(fpath+'processed_data/processed_metadata10.csv')
    arr_test = (np.load(fpath+'processed_data/processed_X_data10.npz')
                        ['arr_0'][1900:2100])
    y_test = (np.load('processed_data/processed_Y_sty_data10.npz')
                        ['arr_0'][1900:2100])

    # Use Standardizing array to fit normalization
    std_arr = (np.load(fpath+'processed_data/standardizing_X_array10.npz')
                        ['arr_0'])
    std = np.expand_dims(std_arr, axis=0)
    # Combine arrays
    arr_test = arr_test.reshape((200, 128, 1292))
    std_and_arr = np.vstack([std,arr_test])
    std_and_arr_scaled = scale_range(std_and_arr,-1,1)

    # After scaling remove stand scaler Array and redimension
    X = std_and_arr_scaled[1:,:,:]
    X_test = X.reshape((200, 128, 1292,1))

    # Shuffle
    X_test = shuffle(X_test, random_state=14)
    y_test = shuffle(y_test, random_state=14)


    return X_test, y_test, df


def crnn(X_test):


    ''' Convolutional Neural Net predictor for Lindy, Balboa, Shag, Charleston


        INPUT:  arr_test - Array of pre-normalized mel-spectrograms
                          (dimensions n x 22) for testing

        OUTPUT: df_pred_crnn - Dataframe of predictions
                        (dimensions m x 4) rows in format:
                        [Lindy, Balboa, Charleston, Shag, display_text]

        '''

    # Load json file and create CRNN Model
    fpath = '/Users/katielazell-fairman/desktop/projects/jazz_music/CNN_Models/'
    with open(fpath+'model-0123024.json', 'r') as json_file:
        loaded_model_json = json_file.read()
        # print loaded_model_json
        crnn_model = model_from_json(loaded_model_json)

    # load weights into new model
    crnn_model.load_weights(fpath+"model-0123024.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    crnn_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


    y_pred_crnn = crnn_model.predict(X_test)
    df_pred_crnn = pd.DataFrame(y_pred_crnn)
    df_pred_crnn.columns = ['Lindy', 'Balboa', 'Charleston', 'Shag']



    return df_pred_crnn

if __name__ == '__main__':
    X_test, y_test, df = load_training_data()
    print "loading data to model..."
    df_pred_crnn = crnn(X_test)

    # Print accuracy score
    y_pred_arr = np.around(np.array(df_pred_crnn),decimals =1)
    accuracy = ((y_test + y_pred_arr) > 1.3).sum()*1./len(y_test)
    print "\n CRNN Model accuracy: {}\n\n".format(round(accuracy,3))

    # Print results (up to first 30)
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
