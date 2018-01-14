import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os


def mel_spectrogram(filename, save_type):
    # Generate Mel-scaled power (energy-squared) spectrogram

    y, sr = librosa.load(filename)
    S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    # Convert to log scale (dB) with peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=150)
    if save_type == 'jpg':
        plt.figure(figsize=(12,4))
        # Display the spectrogram on a mel scale
        librosa.display.specshow(log_S) # optional axis labels sr=sr, x_axis='time', y_axis='mel'
        plt.savefig('{}-MelSpec.jpg'.format(filename[:-4]))
        print '{}-MelSpec.jpg saved'.format(filename[:-4])

    if save_type == 'npy':
        try:
            np.save('{}-MelArr'.format(filename[:-4]), log_S)
            print '{}-MelArr.npy saved '.format(filename[:-4])
        except:
            print "{} did not save".format(filename[:-4])




if __name__ == '__main__':
    csv_list = str(raw_input("enter csv list of filepaths:"))
    start = int(raw_input("enter start row:"))
    stop = int(raw_input("enter end row:"))
    save_type = str(raw_input("would you like to save .jpg or .npy? (jpg/npy):"))
    with open(csv_list) as f:
        for i, wav in enumerate(f):
            if i in range(start, stop):
                if wav.strip()[-4:] == ".wav":
                    try:
                        filename = "wav_samples/" + wav.strip()
                        mel_spectrogram(filename, save_type)
                    except:
                        print "{} did not convert"
