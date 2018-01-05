
import pandas as pd
import numpy as np
import urllib2
import random
import pause
import os
from boto import connect_s3
from StringIO import StringIO


"""
    1. Gets CSV from E3 bucket
    2. Cleans CSV (removes duplicates creates unique file ID names)
    3. Downloads mp3 in batches (batch size <=100)
    4. Transfers batch of mp3s to E3 bucket
    5. Deletes mp3s from local drive
    """


def get_csv_from_s3(bucket,csv_path):

    # Import bash_profile keys
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

    conn =  connect_s3(access_key,secret_key)
    bucket = conn.get_bucket('swingmusic001')
    content = bucket.get_key(csv_path).get_contents_as_string()

    return pd.read_csv(StringIO(content))


def clean_download_list(df_download):


    #Create unique filenames for mp3 (ID & Style)
    df_download['Style-Katie'] = df_download['Style-Katie'].replace("Lindy Hop", "Lindy")
    df_download['filename'] = (df_download['ID'].astype(str)
                                            .str.zfill(4)
                                            + '-' +
                                            df_download['Style-Katie'])
    #Drop duplicate URLs (keep last)
    df_download.drop_duplicates(subset='Link', keep='last', inplace=True)

    return df_download



def download_mp3(df, start, stop):

    #Check batch size is no larger than 100
    if stop-start > 100:
        print "batch is too large"
    else:
        downloaded_mp3s = []
        count = 0
        for mp3, name in zip(df['Link'][start:stop], df['filename'][start:stop]):
            try:
                #Set User-Agent
                req = urllib2.Request(mp3, headers={ 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36' })

                #open mp3 URL for song, download & save
                mp3file = urllib2.urlopen(mp3)
                with open('music_downloads/{}.mp3'.format(name),'wb') as output:
                    output.write(mp3file.read())

                #Add mp3 file name to list
                downloaded_mp3s.append("{}.mp3".format(name))

                #Add to count
                count += 1

                #Pause between 0.3 of a second to 3.5 seconds before next song
                pause.milliseconds(random.randint(300, 3500))

            except:
                #If URL/mp3 reading fails record ID, URL and filename in csv log
                print "{} is not valid :  {}".format(mp3, name)
                with open('error_log/error.txt', 'a') as f:
                    f.write('{},{},{}\n'.format(name[:4],mp3,name))

        print "{} mp3 files downloaded".format(count)
    return downloaded_mp3s

def transfer_to_s3(downloaded_mp3s):
    #Transfer data to s3
    for mp3_file_name in downloaded_mp3s:
        bucket_and_path = 'swingmusic001/music_downloads/{}'.format(mp3_file_name)
        try:
            #transfering to
            os.system('aws s3 cp music_downloads/{} s3://{}'.format(mp3_file_name, bucket_and_path)) # AWS Command line for copying file to S3 bucket
        except:
            #Failed to transfer mp3
            print "Failed to transfer {} to S3".format(mp3_file_name)


def delete_mp3_from_local(downloaded_mp3s):
    for mp3 in downloaded_mp3s:
        os.system("rm music_downloads/{}".format(mp3))
    print "{} mp3 files removed from local".format(len(downloaded_mp3s))


if __name__ == '__main__':
    bucket, csv_path = "swingmusic001", "metadata/Swing_Dance_Style_Master_DownloadC.csv"
    df = get_csv_from_s3(bucket,csv_path)
    df_downloads =  clean_download_list(df)

    cont = 'y'
    start = int(raw_input("Enter start row:"))
    stop = int(raw_input("Enter stop row:"))

    while cont == 'y':
        downloaded_mp3s = download_mp3(df_downloads, start, stop)
        transfer_to_s3(downloaded_mp3s)
        delete_mp3_from_local(downloaded_mp3s)
        cont = raw_input("Continue to download next batch? (y/n)")
        if cont == 'y':
            start = stop
            stop = int(raw_input("Enter new stop row:"))
        else:
            break
