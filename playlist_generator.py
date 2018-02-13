import pandas as pd
import numpy as np
from sklearn.utils import shuffle

'''
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
Generate a playlist under a specified time given a dance style
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

'''

def generate_playlist(time_minutes,Style,df):

    # Conditional Statements
    exclude_poor_quality = ((df['Poor_Quality_Recording'].isnull()) | (df['Poor_Quality_Recording'] == False))
    exclude_not_great = (df['Not_Great'] == 0)
    style_true = (df[Style] == 1)
    non_superlike = (df['Superlike'].isnull() ==True)
    superlike = (df['Superlike'].isnull() ==False)

    # Generate Regular & Superlike DataFrames
    reg = df[exclude_poor_quality & exclude_not_great & style_true & non_superlike]
    superlike = df[exclude_poor_quality & exclude_not_great & style_true & superlike]

    # Concatenate and Shuffle
    playlist = pd.concat([superlike.sample(3), reg.sample(9)])
    playlist = shuffle(playlist)
    playlist['Year'] = playlist['Year'].values.astype(int)

    # Check length
    while playlist['duration'].sum()/60 > time_minutes:
        playlist = playlist[:-1]

    return playlist[['Artist', 'Title', 'Year', 'Link']]


if __name__ == '__main__':
    # Get Metadata CSV file
    df = pd.read_csv('music_downloads/mp3_song_master_for_RF_KNN_model.csv')

    # User playlist input
    time_minutes = 30 #30 minutes default
    Style = str(raw_input("What dance style? (Lindy, Slow_Swing_Blues, Blues, Balboa, Charleston, Shag) :"))

    print generate_playlist(time_minutes,Style,df)
