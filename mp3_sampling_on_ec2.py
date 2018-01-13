
from pydub import AudioSegment, effects
import random
import numpy as np
import pandas as pd
import os



## !!!! Refactor this into a Class




def song_samples(splits,duration,song_mp3,skip=20):

    """
    INPUTS
    Splits: (int) no. of splits, the song will be is split into (where feasible)
    duration: (int) number of seconds of each sample (e.g. 10.5 seconds)
    """

    #def song_stats(splits,duration,song_mp3,skip=30)
    song = AudioSegment.from_file(song_mp3)
    song_duration = (song.duration_seconds - skip)/splits
    # print song.duration_seconds
    # print song_duration

    max_wait = int(round((song.duration_seconds - (splits*duration))/splits,0))
    #print "max wait:",max_wait
    #Return song, max_wait

    if song_duration > duration:
        start = skip

        #def get_splits_save(splits,duration,song_mp3,skip=30, spacer_seconds)

        for split in range(1,splits+1):
            try:
                #Extract segment
                segment = song[start*1000:(start+duration)*1000]
                segment = segment.set_channels(1)
                segment = effects.normalize(segment)

                #create random lag time between segment
                spacer_seconds = random.randint(0, max_wait)
                start = start+duration+spacer_seconds

                print start
                print "spacer: {}".format(spacer_seconds)
                if segment.duration_seconds == duration:

                    filename ="{}-{}.wav".format(song_mp3[:-4],str(split).zfill(2))
                    #Save wav file
                    segment.export("{}".format(filename), format="wav")

                    print filename, "saved"
                #segment.export(filename, format="wav")
                else:
                    segment = song[-duration*1000:]
                    segment = segment.set_channels(1)
                    segment = effects.normalize(segment)
                    print segment.duration_seconds
                    filename ="{}-{}.wav".format(song_mp3[:-4],str(split).zfill(2))
                    #Save wav file
                    segment.export("{}".format(filename), format="wav")

                    print filename, " saved"
                    break

            except:
                print "Unable to process audio file {}".format(song_mp3)
                with open('error.txt', 'wa') as f:
                    f.write('{}\n'.format(song_mp3))
                break

    else:
        print "nope!"
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

def split_rule(df,song_id):
    #If a swing-dancable song is Superliked, take 7 samples
    if (df['Swing_Danceable'].loc[song_id] + df['Superlike_True'].loc[song_id])  == 2:
        return 7
    #If a swing-dancable song is disliked, take 4 samples
    if (df['Swing_Danceable'].loc[song_id] + df['Not_Great'].loc[song_id]) == 2:
        return 4
    #If a swing-dancable song is neither superliked or disliked, take 6 samples
    if (df['Swing_Danceable'].loc[song_id] + df['Not_Great'].loc[song_id]) == 1:
        return 6
    else:
        return 4



if __name__ == '__main__':
    csv_file = str(raw_input("Enter csv_file to process:"))
    df = pd.read(csv_file)
    start = int(raw_input("Enter start row:"))
    stop = int(raw_input("Enter stop row:"))
    os.system("mkdir music_downloads/wav_samples")
    cont = 'y'
    while cont == 'y':

        if stop - start > 200:
            print "batch too large"

        for song_id in range(start,stop):
            song_mp3 = "music_downloads/" + str(df['filename'].loc[song_id]) + ".mp3"
            print song_mp3
            splits = split_rule(df,song_id)
            try:
                song_samples(splits,30,song_mp3,skip=30)
            except:
                print "{} did not convert".format(song_id)


        cont = raw_input("Continue upload next batch? (same directory) (y/n)")
        if cont == 'y':
            start = stop
            stop = str(raw_input("Enter stop row:"))
        else:
            break
