import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os

from ec2_wav_pipeline import create_directories_and_text_files, download_mps, get_list
from download_mp3_ec2 import transfer_df_to_s3


def get_mp3_features(filename, count):
    '''Generates the following for each mp3 song:
        - tempo (harmonic, precussive)
        - beats (harmonic, precussive)
        - root mean square energy per segment (mean, median, std)

    '''
    try:
        # Load audio data
        y, sr = librosa.load(filename)

        # Get Tempo & Beats
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        h_tempo, h_beats = librosa.beat.beat_track(y=y_harmonic, sr=sr)
        p_tempo, p_beats = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # Get Root Mean Squared Energy (avg, median & standard deviation)
        rmse_arr = librosa.feature.rmsa(y=y, sr=sr)
        avg_rmse = rmse_arr.mean()
        med_rmse = np.ma.median(rmse_arr)
        std_rmse = rmse_arr.std()

        # Get length of song
        song = AudioSegment.from_file(filename)
        song_duration = song.duration_seconds


        # Append results to csv
        with open('mp3_audio_features-{}.csv'.format(str(count).zfill(4)), 'a+') as f:
            for feature in [filename[:-4],h_tempo, h_beats, p_tempo,p_beats ,avg_rmse ,med_rmse ,std_rmse, song_duration]:
            f.write("{},".format(feature))
            f.write("/n")

        return y, src

    except:
        print "{} was not able to be analyzed, data not saved".format(filename[:-4])


def save_chroma_data(filename, y, sr, count):

    # Get Pitch percentages of song
    chroma = librosa.feature.chroma_cqt(y=y_2, sr=sr_2)

    #Get chroma & reduce noise by using nearest neighbors with cosine similarity to reduce noise
    chroma_med = librosa.decompose.nn_filter(chroma,
                                     aggregate=np.median,
                                     metric='cosine')
    rec = librosa.segment.recurrence_matrix(chroma, mode='affinity',
                                     metric='cosine', sparse=True)

    chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec,
                                    aggregate=np.average)
    pitch_scale = ['B','B♯', 'A', 'G♯', 'G', 'F♯', 'E', 'D♯', 'D', 'C♯']
    values = []

    for i, pitch in enumerate(pitch_scale):
            values.append(chroma_nlm[i].sum())
    # Return pitch percentage of prominance (sum of values for each pitch over sum of all pitches)
    pitch_values = values / sum(values)

    with open('mp3_audio_features_pitch-{}.csv'.format(str(count).zfill(4)), 'a+') as f:
        for feature in [filename[:-4],h_tempo, h_beats, p_tempo,p_beats ,avg_rmse ,med_rmse ,std_rmse, song_duration]:
        f.write("{},".format(feature))
        f.write("/n")




if __name__ == '__main__':
    csv_list = str(raw_input("enter csv list of filenames:"))

    os.system("touch audio_tempo_info-{}.csv".format(csv_list[6:-3]))
        with open('mp3_audio_features.csv', 'a') as f:
            f.write "filename, h_tempo, h_beats, p_tempo, p_beats, avg_rmse ,med_rmse ,std_rmse"

    # Create directories
    create_directories_and_text_files()

    # Enter download details
    bucket_name = str(raw_input("Enter bucket name:"))
    csv = str(raw_input("Enter csv list to download:")) #in jazz_mmusic directory list of all mp3 files to download from S3
    start = 0
    stop = 3
    cont = 'y'
    count = 1
    while cont == 'y':
        # Download mp3s
        downloaded_mp3s = download_mps(csv, bucket_name, start, stop)

        # Navigate to music_downloads folder and generate list of files
        get_list('music_downloads', 'mp3s.txt')

        #Get Audio Features
        with open('mp3s.txt', 'r') as f:
            for i, filename in enumerate(f):
                while (i => start) & while (i < stop):
                    try:
                        y, src = get_mp3_features(filename, count)
                    except:
                        print "{} did not convert".format(line)
        #Upload Audio Feature csv to cloud & remove from local
        # transfer_df_to_s3(downloaded_mp3s)
        # delete_mp3_from_local(downloaded_mp3s)
        count +=1
        cont = raw_input("Continue to download next batch? (y/n)")
        if cont == 'y':
            start = stop
            stop += 250
        else:
            break
