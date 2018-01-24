import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
from  pydub import AudioSegment
import matplotlib.pyplot as plt
import pickle

from mp3_sampling import song_samples
from CRNN_ensemble_model import scale_range, crnn
from RF_KNN_ensemble_model import check_if_lindy, check_if_blues, k_nearest_neighbor



def download_mp3_youtube(url):
    os.system("cd Demo_testing")
    os.system("youtube-dl --extract-audio --audio-format mp3 {}".format(url))
    os.system("cd ..")

def get_mp3_audio_features(filename):
    '''Generates the following for each mp3 song:
        - tempo (harmonic, precussive)
        - beats (harmonic, precussive)
        - root mean square energy per segment (mean, median, std)
        - song duration
        - pitch prominance values 'B','A#', 'A', 'G#', 'G', 'F#','F', 'E' etc.
        - engineered tempo fearures
    '''
    print "Processing: " + filename + " this may take a while ..."
    # Load audio data
    y, sr = librosa.load(filename)
    print "loading file.."

    # Get Tempo & Beats
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    h_tempo, h_beats = librosa.beat.beat_track(y=y_harmonic, sr=sr)
    p_tempo, p_beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
    tempo_differ = h_tempo - p_tempo
    if (p_tempo <= 120) & (tempo_differ == 0):
        slow_tempo_correction = p_tempo*2
    else:
        slow_tempo_correction = p_tempo

    print slow_tempo_correction
    print "Processed tempo & beats"

    # Get Root Mean Squared Energy (avg, median & standard deviation)
    rmse_arr = librosa.feature.rmse(y=y)
    avg_rmse = rmse_arr.mean()
    med_rmse = np.ma.median(rmse_arr)
    std_rmse = rmse_arr.std()
    print "Procesed RMSEs"

    # Get length of song
    try:
        song = AudioSegment.from_file(filename)
        song_duration = song.duration_seconds
        print "Processed durations"
    except:
        "error getting song duration"
        song_duration = np.NaN

    audio_data = [h_tempo, len(h_beats), p_tempo, len(p_beats),
                  avg_rmse ,med_rmse ,std_rmse, song_duration, tempo_differ,
                  slow_tempo_correction]

    return audio_data, y, sr

def get_mp3_pitch_features(filename, y, sr):
    # Get Pitch percentages of song
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    print "got chroma..."

    # Get chroma & reduce noise by using nearest neighbors with cosine
    # similarity to reduce noise
    chroma_med = librosa.decompose.nn_filter(chroma,
                                             aggregate=np.median,
                                             metric='cosine')
    rec = librosa.segment.recurrence_matrix(chroma, mode='affinity',
                                            metric='cosine', sparse=True)
    chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec,
                                             aggregate=np.average)
    print "got chroma_nlm..."

    pitch_scale = ['B','A#', 'A', 'G#', 'G', 'F#','F', 'E', 'D#', 'D','C#','C']
    values = []

    for i, pitch in enumerate(pitch_scale):
        values.append(chroma_nlm[i].sum())
    # Return pitch percentage of prominance (sum of values for each pitch over
    # sum of all pitches)
    pitch_values = values / sum(values)

    return pitch_values

def create_X_data(audio_data, pitch_values,filename):

    X = np.concatenate((np.array(audio_data), np.array(pitch_values)), axis=0)

    df_X =  pd.DataFrame([X])
    df_X.columns = ['h_tempo', 'h_beats', 'p_tempo', 'p_beats',
                  'rmse mean','rmse median', 'rmse std', 'duration',
                  'tempo-differ','slow-tempo-correction', 'B', 'A#', 'A',
                  'G#', 'G', 'F#', 'F', 'E', 'D#', 'D', 'C#', 'C']
    df_X['filename'] = filename
    return df_X

def mel_spectrogram(filename):
    # Generate Mel-scaled power (energy-squared) spectrogram
    y, sr = librosa.load(filename)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)

    # Convert to log scale (dB) with peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=150)
    plt.figure(figsize=(12,4))
    # Display the spectrogram on a mel scale
    librosa.display.specshow(log_S)
    plt.show()

    try:
        np.save('{}-MelArr'.format(filename[:-4]), log_S)
        print '{}-MelArr.npy saved '.format(filename[:-4])
    except:
        print "{} did not save".format(filename[:-4])
    return log_S

def prep_array(arr):

    # Use Standardizing array to fit normalization
    std_arr = (np.load('processed_data/standardizing_X_array10.npz')
                        ['arr_0'])
    std = np.expand_dims(std_arr, axis=0)
    # Combine arrays
    arr = arr.reshape((1, 128, 1292))
    std_and_arr = np.vstack([std, arr])
    std_and_arr_scaled = scale_range(std_and_arr,-1,1)

    # After scaling remove stand scaler Array and redimension
    X = std_and_arr_scaled[1:,:,:]
    X_arr = X.reshape((1,128, 1292,1))
    print X_arr
    return X_arr

def predict_from_ensamble_model(df_X):
    lindy_rf_model = pickle.load(open('CNN_Models/lindy_rf_model.pkl', 'rb'))
    blues_rf_model = pickle.load(open('CNN_Models/blues_rf_model.pkl', 'rb'))
    knn_model = pickle.load(open('CNN_Models/knn_model.pkl', 'rb'))

    y_pred_lindy = lindy_rf_model.predict(df_X)
    df_pred_lindy = pd.DataFrame(y_pred_lindy)
    df_pred_lindy.columns = ['Lindy_?']

    y_pred_blues = blues_rf_model.predict(df_X)
    df_pred_blues = pd.DataFrame(y_pred_blues)
    df_pred_blues.columns = ['Blues_?']

    y_pred_knn = knn_model.predict(df_X)
    df_pred_knn = pd.DataFrame(y_pred_knn)
    df_pred_knn.columns = ['knn']

    # Ensamble Results Table
    df_results = pd.merge(df_test.reset_index(), y_pred_lindy, how='outer', left_index=True, right_index=True)
    df_results = pd.merge(df_results, y_pred_blues, how='outer', left_index=True, right_index=True)
    df_results = pd.merge(df_results, y_pred_knn, how='outer', left_index=True, right_index=True)

    df_results = df_results[['filename','Dance style', 'Lindy_?', 'Blues_?','knn']]

    df_results['Dance Prediction'] = 0
    for i, (x, y, z) in enumerate(zip(df_results['Lindy_?'], df_results['Blues_?'], df_results['knn'])):
        df_results['Dance Prediction'].iloc[i] = z
        if x == 1:
            df_results['Dance Prediction'].iloc[i] = 'Lindy'
        if y == 1:
            df_results['Dance Prediction'].iloc[i] = 'Blues'
    print df_results['Dance Prediction']


if __name__ == '__main__':
    new = str(raw_input('\nWould you like to sample a new song? (y/n):'))
    if new == 'y':
        url = str(raw_input('Enter url link:'))
        download_mp3_youtube(url)

    filename = "Demo_testing/"+str(raw_input('Locate mp3 and copy filename:')
                                   )
    model = str(raw_input('\nRun ensemble model or neural net? (e/nn):'))
    if model == 'e':
        audio_data, y, sr = get_mp3_audio_features(filename)
        pitch_values = get_mp3_pitch_features(filename, y, sr)
        df_X = create_X_data(audio_data,pitch_values,filename)
        #predict_from_ensamble_model(df_X)



    if model == 'nn':
        wav_filename  = song_samples(1,30,filename,skip=20)
        array = mel_spectrogram(wav_filename)
        array_reshaped = array.reshape((128,1292,1))
        X_arr = prep_array(array_reshaped)
        df_pred_crnn = crnn(X_arr)
        y_pred_arr = np.around(np.array(df_pred_crnn),decimals =1)

        # Print results (up to first 30)
        for i, x in enumerate(y_pred_arr[:30]):
            print "\n\nsong: {}\n".format(wav_filename)
            if df_pred_crnn.iloc[i][0] > 0.25:
                print "Lindy",round(df_pred_crnn.iloc[i][0],2)*100, "%"

            if df_pred_crnn.iloc[i][1] > 0.25:
                print "Balboa ",round(df_pred_crnn.iloc[i][1],2)*100, "%"

            if df_pred_crnn.iloc[i][2] > 0.25:
                print "Shag", round(df_pred_crnn.iloc[i][2],2)*100, "%"

            if df_pred_crnn.iloc[i][3] > 0.25:
                print "Charleston", round(df_pred_crnn.iloc[i][3],2)*100, "%"
            print "\n\n"