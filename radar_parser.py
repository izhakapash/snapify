import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def hann(sample):
    """
    Preformes Hann smoothing of 'iq_sweep_burst'.

    Arguments:
      iq {ndarray} -- 'iq_sweep_burst' array

    returns:
      iq {ndarray}
    """
    filter_weights = np.hanning(sample.shape[0])
    filter_weights = np.expand_dims(filter_weights,1)
    sample = sample * filter_weights
    return sample

def get_spectrogram(sample):
    spectrogram_complex = np.fft.fft((sample), axis=0)
    spectrogram_complex = np.fft.fftshift(spectrogram_complex)
    return spectrogram_complex

def get_db(sample):
    sample_db = 20*np.log(np.abs(sample)+0.000001)
    return sample_db

def get_abs(sample):
    sample_abs = np.abs(sample)
    return sample_abs

def get_phase(sample):
    sample_phase = np.angle(sample)
    return sample_phase

def preprocess(sample):
    spectrogram = get_spectrogram(sample)
    spectrogram_db = get_db(spectrogram)
    spectrogram_phase = get_phase(spectrogram_db)
    return spectrogram ,spectrogram_phase


output_folder = 'C:/Users/izhak/Desktop/atalef/7'
csv_file = 'C:/Users/izhak/Desktop/atalef/7/FileName.csv'
bin_file = "C:/Users/izhak/OneDrive/Desktop/radar_processing/adc_data_Raw_0.bin"



with open("file.bin", "rb") as binary_file:
    # "rb" mode means opening the file in binary read mode
    # "file.bin" is the name of the binary file you want to open

    # do something with the binary file here
    # for example, you can read its contents into a variable:
    binary_data = binary_file.read()

    # or you can iterate over the file contents byte by byte:
    byte = binary_file.read(1)
    while byte:
        # do something with the byte here
        byte = binary_file.read(1)

with open(csv_file) as fp:
    reader = csv.reader(fp, delimiter=",", quotechar='"')
    # next(reader, None)  # skip the headers
    data = [row for row in reader]
tx1_data = []
tx2_data = []
tx3_data = []
tx4_data = []
TX1_data = []
TX2_data = []
TX3_data = []
TX4_data = []
print(data)
for chirp in data:
    for j, signal in enumerate(chirp):
        if j%4 == 0 :
            tx1_data.append((complex(signal.replace("i","j"))))
        if j%4 ==1:
            TX2_data.append((complex(signal.replace("i","j"))))
        if j%4 == 2:
            tx1_data.append((complex(signal.replace("i","j"))))
        if j%4 ==3:
            TX2_data.append((complex(signal.replace("i","j"))))
    TX1_data.append(tx1_data)
    tx1_data =[]
    TX2_data.append(tx2_data)
    tx2_data = []
    TX3_data.append(tx3_data)
    tx3_data = []
    TX4_data.append(tx4_data)
    tx4_data = []

TX1_data = np.asarray(TX1_data)
raw_sample = hann(TX1_data)
spectrogram = get_spectrogram(raw_sample)
plt.plot(spectrogram)
plt.yscale("log")
plt.title("spectrogram")
plt.savefig(output_folder + "/" + 'spectrogram.png')
plt.show()


spectrogram_db = get_db(spectrogram)
plt.plot(spectrogram_db)
plt.title("spectrogram_db")
#plt.yscale("log")
plt.savefig(output_folder + "/" + 'spectrogram_db.png')
plt.show()


spectrogram_phase = get_phase(spectrogram_db)
plt.plot(spectrogram_phase)
plt.title("phase")
plt.savefig(output_folder + "/" + 'spectrogram_phase.png')
plt.show()


