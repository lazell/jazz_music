from boto import connect_s3
import os


def get_bucket_files(bucket):

    # Import bash_profile keys
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

    # Connect to S3
    conn =  connect_s3(access_key,secret_key)
    bucket = conn.get_bucket('swingmusic001')
    for key in bucket:
        print key.name

if __name__ == '__main__':
    bucket = 'swingmusic001'
    get_bucket_files(bucket)
