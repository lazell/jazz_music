
import os


"""Takes file path containing links to mp3 download (from 'music_downloads' if not specified),
    and uploads files beginning with a certain string to the S3 project bucket swingmusic001
    """

def upload_mp3s_to_s3(starts_with_str, file_path="music_downloads"):

    #Upload mp3 files IDs which start with specific string.

    for file in os.listdir(file_path):
        if file.startswith(starts_with_str):
            mp3_file_name = (os.path.join("", file))
            bucket_and_path = 'swingmusic001/music_downloads/{}'.format(mp3_file_name)

            # AWS Command line for copying file to S3 bucket
            os.system('aws s3 cp music_downloads/{} s3://{}'.format(mp3_file_name, bucket_and_path))



if __name__ == '__main__':
    cont = 'y'
    file_path = str(raw_input("File path:"))
    starts_with_str = str(raw_input("Enter file start with string (e.g. '001'):"))

    while cont == 'y':
        upload_mp3s_to_s3(starts_with_str)
        cont = raw_input("Continue upload next batch? (same directory) (y/n)")
        if cont == 'y':
            starts_with_str = str(raw_input("Enter file start with string (e.g. '001'):"))
        else:
            break
