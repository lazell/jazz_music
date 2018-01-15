import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os


#from download_mp3_ec2 import transfer_df_to_s3


def get_mp3_features(filename):
    '''Generates the following for each mp3 song:
        - tempo (harmonic, precussive)
        - beats (harmonic, precussive)
        - root mean square energy per segment (mean, median, std)

    '''
    print "filenam 2:", filename
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
        # with open('mp3_audio_features-{}.csv'.format(str(count).zfill(4)), 'a+') as f:
        #
        #     for feature in [filename[:-4],h_tempo, h_beats, p_tempo,p_beats ,avg_rmse ,med_rmse ,std_rmse, song_duration]:
        #         f.write("{},".format(feature))
        #         f.write("/n")


        # Generate results in dataframe
        cols = ['filename','h_tempo', 'h_beats', 'p_tempo' ,'p_beats' ,'avg_rmse' ,'med_rmse' ,'std_rmse', 'song_duration']
        df_audio_features = pd.DataFrame(columns = cols)
        df_values = pd.DataFrame([filename[:-4],h_tempo, len(h_beats), p_tempo, len(p_beats) ,avg_rmse ,med_rmse ,std_rmse, song_duration]).transpose()
        df_values.columns = cols

        return y, sr, df_values

    except:
        print "{} was not able to be analyzed, data not saved".format(filename[:-4])



def get_chroma_data(filename, y, sr):

    # Get Pitch percentages of song
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)

    #Get chroma & reduce noise by using nearest neighbors with cosine similarity to reduce noise
    chroma_med = librosa.decompose.nn_filter(chroma,
                                     aggregate=np.median,
                                     metric='cosine')
    rec = librosa.segment.recurrence_matrix(chroma, mode='affinity',
                                     metric='cosine', sparse=True)

    chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec,
                                    aggregate=np.average)
    pitch_scale = ['B','B#', 'A', 'G#', 'G', 'F#', 'E', 'D#', 'D', 'C# ']
    values = []

    for i, pitch in enumerate(pitch_scale):
            values.append(chroma_nlm[i].sum())
    # Return pitch percentage of prominance (sum of values for each pitch over sum of all pitches)
    pitch_values = values / sum(values)

    # Generate Dataframe of pitch features for song
    df_pitch = pd.DataFrame(pitch_values).transpose()
    df_pitch.columns = pitch_scale
    df_pitch['filename'] = None
    df_pitch['chroma_arr'] = None
    df_pitch['filename'].iloc[0] = filename
    df_pitch['chroma_arr'].iloc[0] = chroma_nlm

    return df_pitch

def download_mp3s(csv, bucket_name, start, stop):
    with open(csv) as f:
        for i, line in enumerate(f):
            if i in range(start,stop):
                os.system('aws s3 cp s3://{}/music_downloads/{} {}'.format(bucket_name, line.strip(), line.strip()))



if __name__ == '__main__':

    count = 1

    # Enter download details
    bucket_name = str(raw_input("Enter bucket name:"))
    csv_list = str(raw_input("Enter csv list to download:")) #in jazz_mmusic directory list of all mp3 files to download from S3
    start = 0
    stop = 3

    cont = 'y'
    while cont == 'y':
        # Download mp3s
        downloaded_mp3s = download_mp3s(csv_list, bucket_name, start, stop)

        #Initialize dataframe
        chroma_cols = ['filename','B','B#', 'A', 'G#', 'G', 'F#', 'E', 'D#', 'D', 'C# ','chroma_arr']
        df_c = pd.DataFrame(columns=chroma_cols)

        #Get Audio Features
        with open('mp3s.txt', 'r') as f:
            for i, filename in enumerate(f):
                print "filename:", filename
                if (i >= start) & (i < stop):
                    try:
                        y, sr, df_audio_features= get_mp3_features(musfilename)
                        print "mp3 featues work"
                        df_pitch = get_chroma_data(filename, y, sr)
                        print " chroma works"
                        #Add data to dataframe
                        df_c = df_c.append(df_pitch)
                        df_c = df_c.merge(df_values, on='filename')

                    except:
                        print "{} did not convert".format(filename)



        # Collate & Save DataFrame
        df_c.to_pickle('mp3_audio_features_{}.pkl'.format(str(count).zfill(4)))

        # Upload Audio Feature csv to cloud & remove from local
        os.system('aws s3 cp mp3_audio_features_{}.pkl s3://{}/processed_data/mp3_audio_features_{}.pkl'.format(str(count).zfill(4), bucket_name, str(count).zfill(4)))
        # delete_mp3_from_local(downloaded_mp3s)
        #os.system('rm music_downloads')

        count +=1
        cont = raw_input("Continue to download next batch? (y/n)")
        if cont == 'y':
            start = stop
            #stop += 250
        else:
            break
