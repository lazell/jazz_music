import os
#navigating from jazz_music
import mp3_sampling as mp3

def create_directories_and_text_files():
    os.system('mkdir music_downloads')



def download_mps(csv, bucket_name, start, stop):
    with open(csv) as f:
        for i, line in enumerate(f):
            if i in range(start,stop):
                os.system('aws s3 cp s3://{}/music_downloads/{}music_downloads/{}'.format(bucket_name,line.strip(), line.strip()))

def get_list(folder,list_filename):
    # Navigate to  folder and get list of files
    os.system('cd {}'.format(folder))
    os.system('ls > {}'.format(list_filename))

def run_mp3_sampling(csv_file, start, stop):
        df = mp3.create_feature_cols(csv_file)
        os.system("mkdir music_downloads/wav_samples")

        for song_id in range(start,stop):
            song_mp3 = "music_downloads/" + str(df['filename'].loc[song_id]) + ".mp3"
            print song_mp3
            splits = mp3.split_rule(df,song_id)
            print "No. of samples:".format(splits)
            try:
                mp3.song_samples(splits,30,song_mp3,skip=30)
            except:
                print "{} did not convert".format(song_id)


if __name__ == '__main__':
    # Create directories
    create_directories_and_text_files()

    #Download mp3s
    bucket_name = str(raw_input("Enter bucket name:"))
    csv = 'mp3_files.txt' #in jazz_mmusic directory list of all mp3 files to download from S3
    start = int(raw_input("start row:"))
    stop = int(raw_input("stop row:"))

    # Download mp3s
    download_mps(csv, bucket_name, start, stop)

    # Navigate to music_downloads folder and generate list of files
    get_list('music_downloads', 'wav_files_for_s3_transfer.txt')

    #Generate wav files
    csv_file = str(raw_input("Enter csv_file (dataframe metadata) to process:"))
    run_mp3_sampling(csv_file, start, stop)
