
from pydub import AudioSegment, effects
import random
import numpy as np
import pandas as pd
import os


def create_feature_cols(csv_file):

    df = pd.read_csv(csv_file)

    # Determine Superlikes
    yes = df['Superlike'].unique()[1:]
    df['Superlike_True'] = df['Superlike'].map(lambda x: 1 if x in yes else 0)

    # Determine Danceability & Swing-Dancablity
    df['Swing_Danceable'] = df['Style-Katie'].map(lambda x: 1 if x != 'None' else 0)
    df['Other_Danceable'] = df['Notes'].map(lambda x: 1 if x in ['other',
                                        'latin', 'bluegrass', 'country'] else 0)
    df['Danceable'] = df['Swing_Danceable']+ df['Other_Danceable']

    # Identify swing-dancible but not so great songs

    df['Not_Great'] = df['Notes'].map(lambda x: 1 if x in ['not great',
                                                           'not good'] else 0)
    df['Poor_Quality_Recording'] = df['Notes'].str.contains('quality','Quality')

    # Drop NaN
    df = df.dropna(subset=['ID'])

    # Generate filename
    df['IDx'] = df['ID'].map(lambda x: int(x))
    df['Style-Katie'] = df['Style-Katie'].replace('Lindy Hop', 'Lindy')
    df['filename'] = (df['IDx'].astype(str)
                              .str.zfill(4)
                              + '-' +
                              df['Style-Katie'])
    print df.head()
    return df



def song_samples(splits, duration, song_mp3, skip=20):

    '''
    Generates samples based duration and number of splits per song
    (sometimes with random overlap)

    INPUTS:
        - splits   : (int) no. of splits the song will be is split into
        - duration : (int) number of seconds of each sample
        - song_mp3 : filepath of mp3 (.mp3 pr .wav format)
        - skip     : (int) numer of seconds to skip at intro of song

    OUTPUTS:
        - splits   : (int) no. of splits the song will be is split into
        - duration : (int) number of seconds of each sample
        - song_mp3 : filepath of mp3 (.mp3 pr .wav format)
        - skip     : (int) numer of seconds to skip at intro of song

    '''

    song = AudioSegment.from_file(song_mp3)
    song_duration = (song.duration_seconds-skip)/splits

    max_wait = int(round((song.duration_seconds-(splits*duration))/splits, 0))

    if song_duration > duration:
        start = skip

        for split in range(1, splits+1):
            try:
                # Extract segment
                segment = song[start*1000:(start+duration)*1000]
                segment = segment.set_channels(1)
                segment = effects.normalize(segment)

                # create random lag time between segment
                spacer_seconds = random.randint(0, max_wait)
                start = start+duration+spacer_seconds

                print start
                print "spacer: {}".format(spacer_seconds)
                if segment.duration_seconds == duration:

                    filename ="{}-{}.wav".format(song_mp3[:-4],
                                                 str(split).zfill(2))
                    # Save wav file
                    segment.export("{}".format(filename), format="wav")

                    print filename, "saved"
                else:
                    segment = song[-duration*1000:]
                    segment = segment.set_channels(1)
                    segment = effects.normalize(segment)
                    print segment.duration_seconds
                    filename ="{}-{}.wav".format(song_mp3[:-4],
                                                 str(split).zfill(2))
                    # Save wav file
                    segment.export("{}".format(filename), format="wav")

                    print filename, " saved"
                    break

            except:
                print "Unable to process audio file {}".format(song_mp3)
                with open('error.txt', 'wa') as f:
                    f.write('{}\n'.format(song_mp3))
                break

    else:
        start = skip

        for split in range(1,splits+1):
            #Extract segment
            segment = song[start*1000:(start+duration)*1000]
            segment = segment.set_channels(1)
            segment = effects.normalize(segment)

            #create random lag time between segment
            spacer_seconds = random.randint(-10, 0)
            start = start+duration+spacer_seconds

            print start
            print "spacer: {}".format(spacer_seconds)
            if segment.duration_seconds == duration:

                filename ="{}-{}.wav".format(song_mp3[:-4],str(split).zfill(2))
                print filename, " saved"
                #Save wav file
                segment.export("{}".format(filename), format="wav")

            else:
                segment = song[-duration*1000:]
                segment = segment.set_channels(1)
                segment = effects.normalize(segment)
                #print segment.duration_seconds
                filename ="{}-{}.wav".format(song_mp3[:-4],str(split).zfill(2))
                #Save wav file
                segment.export("{}".format(filename), format="wav")

                print filename, " saved"
                break
    return filename

def split_rule(df,song_id):
    # If a swing-dancable song is Superliked, take 7 samples
    if (df['Swing_Danceable'].loc[song_id]+df['Superlike_True'].loc[song_id])==2:
        return 7
    # If a swing-dancable song is disliked, take 4 samples
    if (df['Swing_Danceable'].loc[song_id]+df['Not_Great'].loc[song_id]) == 2:
        return 4
    # If a swing-dancable song is not superliked or disliked, take 6 samples
    if (df['Swing_Danceable'].loc[song_id]+df['Not_Great'].loc[song_id]) == 1:
        return 6
    else:
        return 4



if __name__ == '__main__':
    csv_file = str(raw_input("Enter csv_file to process:"))
    df = create_feature_cols(csv_file)

    start = int(raw_input("Enter start row:"))
    stop = int(raw_input("Enter stop row:"))
    os.system("mkdir music_downloads/wav_samples")

    cont = 'y'
    while cont == 'y':
        # Check batch processing size
        if stop - start > 200:
            print "batch too large"

        # Split-sample each song in batch
        for song_id in range(start, stop):
            song_mp3 = 'music_downloads/'+str(df['filename'].loc[song_id])+'.mp3'
            print song_mp3
            splits = split_rule(df,song_id)
            print splits
            try:
                song_samples(splits, 30, song_mp3, skip=20)
            except:
                print "{} did not convert".format(song_id)


        cont = raw_input("Continue upload next batch? (same directory) (y/n)")
        if cont == 'y':
            start = stop
            stop = str(raw_input("Enter stop row:"))
        else:
            break
