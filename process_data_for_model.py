import pandas as pd
import numpy as np
import csv
import os
import re
from scipy.misc import imresize
from ec2_download_data import get_npy_from_s3


def list_melspec():
    # Create List of Mel-Spec array files
    os.system("python get_bucket_files.py > s3_files.txt")
    with open("s3_files.txt","r") as f:
        for line in f:
            with open("mel_spec_files.csv", "w") as arrays:
                if line[-10:] == "MelArr.npy":
                    arrays.write(line)

def download_melspecs():
    with open("mel_spec_files.csv", "w") as arrays:
        for mel-spec in array:
            bucket = 'swingmusic001'
            file_path = mel-spec
            file_path = filepath.split("/")[:-1]
            dir_ = "/".join(file_path)
            get_npy_from_s3(bucket,file_path, dir_):


def get_3D_array_resize(arr_name):
    #Open nupy numpy array of mel-spec
    arr = np.load(arr_name)
    #arr = imresize(arr, 0.6)
    #Convert to 3D array - (N,M,1) shape
    arr = arr.reshape((arr.shape[0],arr.shape[1],1))
    return arr


def create_subset_dataframe():
    # Create merged dataframe (matching arrays to their metadata)
    df_melspec = pd.read_csv('mel_spec_files.csv', header=None)

    #Add filename reference to dataframe
    df_melspec['filename'] = df_melspec[0].map(lambda x: x[:-14])
    df_arr = np.array(df_micro['mel-spec-array']
    arr = np.stack(df_arr)

    df_subset = df_melspec.merge(df,how='left', on='filename')
    df_subset['mel-spec-array'] = df_subset['3d_arr'].apply(get_3D_array_resize)

    return df_subset

def create_X_and_Y_arrays(df,X_col_label, Y_col_label):
    # Generate stacked array of mel-spec data
    df_arr = np.array(df[X_col_label])
    arr = np.stack(df_arr)

    # Save transformed array data
    np.save('processed_X_data.npy', arr)

    # Save array of labels
    np.save('processed_Y_data.npy', np.array(df[Y_col_label]))

if __name__ == '__main__':
    #Get Metadata CSV file
    df = pd.read_csv(str(raw_input("Enter csv file path:"))
    print df.describe(), "/n"
    print df.head(15)

    list_melspec()
    download_melspecs()
    df = create_subset_dataframe()

    X_col_label = 'mel-spec-array'     #'mel-spec-array'
    Y_col_label = 'Style-Katie'        #'Style-Katie'

    create_X_and_Y_arrays(df,X_col_label, Y_col_label)
