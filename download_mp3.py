
import pandas as pd
import numpy as np
import urllib2
import random
import pause

"""Takes csv file containing links to mp3 download (from 'music_downloads'),
    filters for duplicate links, generates a uniqueID filename
    and saves mp3s in 'music_downloads' directory
    Also returns a loist of dead links to error.txt file"""



def clean_download_list(csv_file):

    #Load the file
    df_download = pd.read_csv('music_downloads/'+str(csv_file))

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

    #Check batch size is no larger than 50
    if stop-start > 50:
        print "batch is too large"
    else:
        count = 0
        for mp3, name in zip(df['Link'][start:stop], df['filename'][start:stop]):
            try:
                #Set User-Agent
                req = urllib2.Request(mp3, headers={ 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.94 Safari/537.36' })

                #open mp3 URL for song, download & save
                mp3file = urllib2.urlopen(mp3)
                with open('music_downloads/{}.mp3'.format(name),'wb') as output:
                    output.write(mp3file.read())

                #Add to count
                count += 1

                #Pause between 0.3 of a second to 3.5 seconds before next song
                pause.milliseconds(random.randint(300, 3500))

            except:
                #If URL/mp3 reading fails record ID, URL and filename in csv log
                print "{} is not valid :  {}".format(mp3, name)
                with open('error.txt', 'a') as f:
                    f.write('{},{},{}\n'.format(name[:4],mp3,name))

        print "{} mp3 files downloaded to 'music_downloads'".format(count)

if __name__ == '__main__':
    csv_file = raw_input("Enter csv filename :")
    df_downloads =  clean_download_list(csv_file)

    cont = 'y'
    start = int(raw_input("Enter start row:"))
    stop = int(raw_input("Enter stop row:"))

    while cont == 'y':
        download_mp3(df_downloads, start, stop)
        cont = raw_input("Continue to download next batch? (y/n)")
        if cont == 'y':
            start = stop
            stop = int(raw_input("Enter new stop row:"))
        else:
            break
