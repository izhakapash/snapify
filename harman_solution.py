import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft
from scipy.signal import periodogram


output_folder = "C:/Users/izhak/OneDrive/Desktop/uotput"
radar_file = 'C:/Users/izhak/OneDrive/Desktop/record1.txt'

data = np.genfromtxt(radar_file, delimiter=',')
# Load the signal data
I =  data[:,0]
Q =  data[:,1]



# Preprocess the signal
I = I - np.mean(I) # remove mean value
I = I * np.hamming(len(I)) # apply Hamming window
Q = Q - np.mean(Q) # remove mean value
Q = Q * np.hamming(len(Q)) # apply Hamming window

# Define the cutoff frequency and the order of the Butterworth filter
cutoff_frequency = 2 # Hz
order = 5

# Apply the Butterworth low-pass filter to the signal
b, a = signal.butter(order, cutoff_frequency / (1000 / 2), 'low')
I_filtered = signal.filtfilt(b, a, I)
Q_filtered = signal.filtfilt(b, a, Q)

# Compute the PSD spectrum of the filtered signals
f_I, psd_I = periodogram(I_filtered, fs=1000)
f_Q, psd_Q = periodogram(Q_filtered, fs=1000)

# Plot the PSD spectrum of the filtered signals
plt.plot(f_I, psd_I)
plt.plot(f_Q, psd_Q)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2/Hz)')
plt.title('PSD spectrum of filtered I and Q signals')
plt.xlim(0, 5)
plt.show()

