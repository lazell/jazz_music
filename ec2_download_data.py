
import pandas as pd
import numpy as np
import requests
import random
import pause
import os
from boto import connect_s3
from io import StringIO

def get_npy_from_s3(bucket,file_path, dir_):

    # Import bash_profile keys
    access_key = os.environ['AWS_ACCESS_KEY_ID']
    secret_key = os.environ['AWS_SECRET_ACCESS_KEY']

    try:
        os.system("mkdir {}".format(dir_))
    except:
        continue
    conn =  connect_s3(access_key,secret_key)
    bucket = conn.get_bucket(bucket)
    try:
        content = bucket.get_key(file_path).get_contents_to_filename(file_path)
        print "{} downloaded to local directory".format(file_path)
    except:
        print "Error: could not download check filepath"

if __name__ == '__main__':
    bucket = str(raw_input("Enter bucket name:"))
    file_path = str(raw_input("Enter filepath:"))
    file_path = filepath.split("/")[:-1]
    dir_ = "/".join(file_path)
    get_npy_from_s3(bucket,file_path)
