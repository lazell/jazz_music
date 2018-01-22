import os
import numpy as np
import pandas as pd
import librosa
from  pydub import AudioSegment



def download_mp3_youtube(url, video_name):
    os.system("youtube-dl --extract-audio --audio-format mp3 {} {}".format(url, video_name))
    return'{}-{}.mp3'.format(video_name,url[-11:])


def get_mp3_audio_features(filename):
    '''Generates the following for each mp3 song:
        - tempo (harmonic, precussive)
        - beats (harmonic, precussive)
        - root mean square energy per segment (mean, median, std)
        - song duration
        - pitch prominance values 'B','A#', 'A', 'G#', 'G', 'F#','F', 'E', 'D#', 'D','C#','C'
    '''
    print "Processing: " + filename + " this may take a while ..."
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
    try:
        song = AudioSegment.from_file(filename)
        song_duration = song.duration_seconds
        print "Processed durations"
    except:
        "error getting song duration"
        song_duration = np.NaN

    audio_data = [filename[:-4],h_tempo, len(h_beats), p_tempo, len(p_beats),avg_rmse ,med_rmse ,std_rmse, song_duration]

    return audio_data

def get_mp3_pitch_features(filename):
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

def create_X_data(audio_data, pitch_data):
    X = audio_data  + pitch_values
    return X

if __name__ == '__main__':
