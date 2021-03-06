import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os
import datetime

#from download_mp3_ec2 import transfer_df_to_s3


def get_mp3_features(filename, count):
    '''Generates the following for each mp3 song:
        - tempo (harmonic, precussive)
        - beats (harmonic, precussive)
        - root mean square energy per segment (mean, median, std)

    '''
    print "Processing: " + filename + " this may take a while ..."
    try:
        # Load audio data
        y, sr = librosa.load(filename)
        print "loading file.."

        # Get Tempo & Beats
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        h_tempo, h_beats = librosa.beat.beat_track(y=y_harmonic, sr=sr)
        p_tempo, p_beats = librosa.beat.beat_track(y=y_percussive, sr=sr)
        print "Processed tempo & beats"

        # Get Root Mean Squared Energy (avg, median & standard deviation)
        rmse_arr = librosa.feature.rmse(y=y)
        avg_rmse = rmse_arr.mean()
        med_rmse = np.ma.median(rmse_arr)
        std_rmse = rmse_arr.std()
        print "Procesed RMSEs"

        # Get length of song
        song = AudioSegment.from_file(filename)
        song_duration = song.duration_seconds
        print "Processed durations"
        # Append results to csv
        with open('mp3_audio_features-{}.csv'.format(str(count).zfill(4)), 'a+') as f:
            for feature in [filename[:-4],h_tempo, len(h_beats), p_tempo, len(p_beats) ,avg_rmse ,med_rmse ,std_rmse, song_duration]:
                f.write("{},".format(feature))
            f.write("\n")

        # Generate results in dataframe
        #cols = ['filename','h_tempo', 'h_beats', 'p_tempo' ,'p_beats' ,'avg_rmse' ,'med_rmse' ,'std_rmse', 'song_duration']
        #df_audio_features = pd.DataFrame(columns = cols)
        #df_values = pd.DataFrame([filename[:-4],h_tempo, len(h_beats), p_tempo, len(p_beats) ,avg_rmse ,med_rmse ,std_rmse, song_duration]).transpose()
        #df_values.columns = cols

        print "Mp3 features work!"

        return y, sr
    except:
        print "{} was not able to be analyzed, data not saved".format(filename)



def get_chroma_data(filename, y, sr, count):
    print y
    try:
        # Get Pitch percentages of song
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        print "chroma works"

        #Get chroma & reduce noise by using nearest neighbors with cosine similarity to reduce noise
        chroma_med = librosa.decompose.nn_filter(chroma,
                                         aggregate=np.median,
                                         metric='cosine')
        rec = librosa.segment.recurrence_matrix(chroma, mode='affinity',
                                         metric='cosine', sparse=True)

        chroma_nlm = librosa.decompose.nn_filter(chroma, rec=rec,
                                        aggregate=np.average)
        print "chroma nlm works"

        pitch_scale = ['B','A#', 'A', 'G#', 'G', 'F#','F', 'E', 'D#', 'D','C#','C']
        values = []

        for i, pitch in enumerate(pitch_scale):
                values.append(chroma_nlm[i].sum())
        # Return pitch percentage of prominance (sum of values for each pitch over sum of all pitches)
        pitch_values = values / sum(values)

        # Generate Dataframe of pitch features for song
        #df_pitch = pd.DataFrame(pitch_values) #.transpose()
        #df_pitch.columns = pitch_scale
        #df_pitch['filename'] = None
        #df_pitch['chroma_arr'] = None
        #df_pitch['filename'].iloc[0] = filename
        #df_pitch['chroma_arr'].iloc[0] = chroma_nlm

        #df_pitch.to_pickle('mp3_audio_features-{}.pkl')

        # Append results to csv
        with open('mp3_pitch_features-{}.csv'.format(str(count).zfill(4)), 'a+') as f:
            f.write("{},".format(filename.strip()))
            for value in pitch_values:
                f.write("{},".format(value))
            f.write("\n")

        return df_pitch

    except:
        print "Could not extract pitch features"

def download_mp3s(csv, bucket_name, start, stop):
    lst = []
    with open(csv, 'r') as f:
        for i, line in enumerate(f):
            if i in range(start,stop):
                print "{}".format(line.strip())
                os.system('aws s3 cp s3://{}/music_downloads/{} {}'.format(bucket_name, line.strip(), line.strip()))
                lst.append(line.strip())
    return lst



if __name__ == '__main__':

    count = int(raw_input("file ID format 1: "))

    # Enter download details
    #bucket_name = str(raw_input("Enter bucket name:"))
    #csv_list = str(raw_input("Enter csv list to download:")) #in jazz_mmusic directory list of all mp3 files to download from S3
    bucket_name = "swingmusic001"
    csv_list = "mp3s.txt"
    start = int(raw_input("Start ID: "))
    stop = int(raw_input("Stop ID: "))

    cont = 'y'

    while cont == 'y':
        # Download mp3s
        downloaded_mp3s = download_mp3s(csv_list, bucket_name, start, stop)
        print downloaded_mp3s

        #Initialize pitch dataframe
        #chroma_cols = ['filename','B','A#', 'A', 'G#', 'G', 'F#','F', 'E', 'D#', 'D','C#','C','chroma_arr']
        #df_c = pd.DataFrame(columns=chroma_cols)


        with open('mp3_pitch_features-{}.csv'.format(str(count).zfill(4)), 'a+') as f:
            f.write('filename, B, A#, A, G#, G, F#, F, E, D#, D, C#, C\n')

        with open('mp3_audio_features-{}.csv'.format(str(count).zfill(4)), 'a+') as f:
            f.write('filename, h_tempo, p_tempo, avg_rmse, med_rmse, std_rmse,song_duration\n')

        #Get Audio Features
        with open(csv_list, 'r') as f:
            for i, filename in enumerate(f):
                if (i in range(start,stop)):
                    if (os.stat(filename.strip()).st_size > 130000): #Check if file is in range & larger than 130000
                        print "Attempting feature extract for :", filename
                        try:
                            y, sr = get_mp3_features(filename.strip())
                            get_chroma_data(filename.strip(), y, sr, count)
                            print "Chroma features fetched!"

                            # Collate & Save DataFrame
                            #df_c = df_c.append(df_pitch)
                            #df_c = df_a.merge(df_values, on='filename')
                            print "Dataframe created"

                            print "Dataframe saved! \n"
                        except:
                            continue
                    else:
                        continue

        #df_c.to_pickle('mp3_audio_features-{}.pkl')
        # Upload Audio Feature csv to cloud & remove from local
        os.system('aws s3 cp mp3_audio_features-{}.csv s3://{}/processed_data/mp3_audio_features-{}.csv'.format(str(count).zfill(4), bucket_name, str(count).zfill(4)))
        os.system('aws s3 cp mp3_pitch_features-{}.csv s3://{}/processed_data/mp3_pitch_features-{}.csv'.format(str(count).zfill(4), bucket_name, str(count).zfill(4)))

        # Delete_mp3_from_local(downloaded_mp3s)
        for mp3 in downloaded_mp3s:
            print mp3
            os.system('rm {}'.format(mp3))

        cont = raw_input("Continue to download next batch? (y/n)")
        if cont == 'y':
            start = stop
            stop = str(raw_input("Stop ID: "))
