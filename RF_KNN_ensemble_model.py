import pandas as pd
import numpy as np

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import pickle


def validation_test_split_test_train(df):
    ''' Removes 'Nones' from a train/validation dataframe and returns a
    shuffled train and test dataframes '''

    df = df[df['Dance style'] != 'None']
    splits = ShuffleSplit(n_splits=1, test_size=.2, random_state=9)
    for train_idx, test_idx in splits.split(df):
        print train_idx.shape, test_idx.shape
        return df.iloc[train_idx], df.iloc[test_idx]

def check_if_lindy(df_train, df_test):


    ''' Check if audio is Lindy Hop via
        Lindy Hop Optimized Random Forest Classifier Model

        INPUT:  df_train - Dataframe of processed audio features
                          (dimensions n x 22) for training

                df_test - Dataframe of processed audio features to test
                        (dimensions m x 22) for testing

        OUTPUT: df_pred_lindy - Dataframe of predictions
                        (dimensions m x 1) in format 1 = Lindy, 0 = not lindy

    '''

    lindy_cols = ['h_tempo', 'h_beats', 'p_tempo', 'p_beats',
                  'tempo-differ', 'slow-tempo-correction', 'rmse mean',
                  'rmse median', 'rmse std', 'B', 'A#', 'A', 'G#', 'G',
                  'F#', 'F', 'E', 'D#', 'D', 'C#', 'C']

    X_train = df_train[lindy_cols]
    y_train = (df_train['Dance style'].map(lambda x: 1 if x == 'Lindy' else 0)
                                      .reshape(-1 ,1))
    X_test = df_test[lindy_cols]

    lindy_rf_model = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
                max_depth=5, max_features=13, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=7, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)


    lindy_rf_model.fit(X_train, y_train)

    y_pred_lindy = lindy_rf_model.predict(X_test)
    df_pred_lindy = pd.DataFrame(y_pred_lindy)
    df_pred_lindy.columns = ['Lindy_?']

    pickle.dump(lindy_rf_model,open('CNN_Models/lindy_rf_model.pkl','wb'))
    return df_pred_lindy


def check_if_blues(df_train,df_test):

    ''' Check if Blues via
        Blues Optimized Random Forest Classifier Model

        INPUT:  df_train - Dataframe of processed audio features
                          (dimensions n x 22) for training

                df_test - Dataframe of processed audio features to test
                          (dimensions m x 22) for testing

        OUTPUT: df_pred_blues - Dataframe of predictions
                        (dimensions m x 1) in format 1 = Blues, 0 = not Blues
        '''

    blues_cols = ['h_tempo', 'h_beats', 'p_tempo', 'p_beats',
                  'tempo-differ','duration','slow-tempo-correction', 'rmse mean',
                  'rmse median', 'rmse std', 'B', 'A#', 'A', 'G#', 'G',
                  'F#', 'F', 'E', 'D#', 'D', 'C#', 'C']

    X_train = df_train[blues_cols]
    y_train = (df_train['Dance style'].map(lambda x: 1 if x == 'Blues' else 0)
                                      .reshape(-1 ,1))
    X_test = df_test[blues_cols]

    blues_rf_model = RandomForestClassifier(bootstrap=True, criterion='gini',
                max_depth=None, max_features=13, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_impurity_split=None,
                min_samples_leaf=3, min_samples_split=2,
                min_weight_fraction_leaf=0.0, n_estimators=30, n_jobs=1,
                oob_score=False, random_state=0, verbose=0, warm_start=False)

    blues_rf_model.fit(X_train, y_train)
    y_pred_blues = blues_rf_model.predict(X_test)
    df_pred_blues = pd.DataFrame(y_pred_blues)
    df_pred_blues.columns = ['Blues_?']

    pickle.dump(blues_rf_model,open('CNN_Models/blues_rf_model.pkl','wb'))
    return df_pred_blues

def k_nearest_neighbor(df_train,df_test):

    ''' K Nearest Neighbor Classifier Model

        INPUT:  df_train - Dataframe of processed audio features
                          (dimensions n x 22) for training

                df_test - Dataframe of processed audio features to test
                          (dimensions m x 22) for testing

        OUTPUT: df_pred_knn - Dataframe of predictions
                        (dimensions m x 1)
        '''

    knn_cols = ['h_tempo', 'h_beats', 'p_tempo', 'p_beats', 'rmse mean',
                  'rmse median', 'rmse std', 'B', 'A#', 'A', 'G#', 'G',
                  'F#', 'F', 'E', 'D#', 'D', 'C#', 'C']

    X_train = df_train[knn_cols]
    y_train = (df_train['Dance style'])
    X_test = df_test[knn_cols]

    knn_model = KNeighborsClassifier(n_neighbors=7)
    knn_model.fit(X_train, y_train)

    y_pred_knn = knn_model.predict(X_test)
    df_pred_knn = pd.DataFrame(y_pred_knn)
    df_pred_knn.columns = ['knn']

    pickle.dump(knn_model,open('CNN_Models/knn_model.pkl','wb'))

    return df_pred_knn

if __name__ == '__main__':
    # Get Training/testing data
    df = pd.read_csv('music_downloads/mp3_song_master_for_RF_KNN_model.csv')

    df_train, df_test = validation_test_split_test_train(df)
    print "Train shape: {} Test shape: {}".format(df_train.shape, df_test.shape)

    y_pred_lindy = check_if_lindy(df_train, df_test)
    y_pred_blues = check_if_blues(df_train, df_test)
    y_pred_knn = k_nearest_neighbor(df_train, df_test)

    # Ensamble Results Table
    df_results = pd.merge(df_test.reset_index(), y_pred_lindy, how='outer', left_index=True, right_index=True)
    df_results = pd.merge(df_results, y_pred_blues, how='outer', left_index=True, right_index=True)
    df_results = pd.merge(df_results, y_pred_knn, how='outer', left_index=True, right_index=True)
    df_results = df_results[['filename','Dance style', 'Lindy_?', 'Blues_?','knn']]

    df_results['Correct Prediction'] = 0
    df_results['Dance Prediction'] = 0
    for i, (x, y, z, a) in enumerate(zip(df_results['Dance style'], df_results['Lindy_?'], df_results['Blues_?'], df_results['knn'])):
        # Check Accuracy
        if (x == 'Lindy') &  (y == 1):
            df_results['Correct Prediction'].iloc[i] = 1
        if (x == 'Blues') &  (z == 1):
            df_results['Correct Prediction'].iloc[i] = 1
        if x == a:
            df_results['Correct Prediction'].iloc[i] = 1

        # Dance Prediction
        df_results['Dance Prediction'].iloc[i] = a
        if y == 1:
            df_results['Dance Prediction'].iloc[i] = 'Lindy'
        if z == 1:
            df_results['Dance Prediction'].iloc[i] = 'Blues'


    print df_results.describe()
    print "Example predictions:"
    print df_results[['filename','Dance Prediction']][:30]
