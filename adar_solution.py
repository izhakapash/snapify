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
azimuth = np.arctan(Q/I)
plt.plot(azimuth)
#plt.plot(f_Q, psd_Q)
plt.xlabel('Time (s)')
plt.ylabel('Power spectral density (V^2/Hz)')
plt.title('atan I and Q signals')
#plt.xlim(0, 5)
plt.show()
# Extract the less noisy portion of the signal
less_noisy_signal = azimuth[120000:140000]

# Estimate the PSD using the periodogram function
freq, psd = signal.periodogram(less_noisy_signal, fs=1000)

# Plot the PSD
plt.plot(freq, psd)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power spectral density (V^2/Hz)')
plt.title('PSD of less noisy portion of signal')
plt.xlim(0,5)
plt.show()

fs = 1000
# Compute the FFT of the signal
complex_signal = I + 1j * Q

# Compute the FFT of the complex signal
fft_signal = np.fft.fftshift(np.fft.fft(complex_signal))

# Compute the power spectral density (PSD)
psd = (1.0 / len(I)) * np.square(np.abs(fft_signal))

# Compute the frequency vector
freq = np.fft.fftfreq(len(I), 1.0/1000)

# Plot the PSD
plt.plot(freq[:len(I)//2], psd[:len(I)//2])
#plt.xlim(0,1000)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('FFT of Radar Signal')
plt.show()

# Identify the dominant frequency component corresponding to heart rate
max_index = np.argmax(psd[:len(I)//2])
heart_rate = fs/2 -freq[max_index]

# Print the estimated heart rate
print('Estimated heart rate:', heart_rate, 'Hz')


# Define the filter parameters
fs = 1000
f1 = 0.5
f2 = 5
order = 4
b, a = signal.butter(order, [f1 / (fs / 2), f2 / (fs / 2)], 'bandpass')

# Apply the filter to the signal
I_filt = signal.filtfilt(b, a, I)
Q_filt = signal.filtfilt(b, a, Q)

# Compute the FFT of the signal
complex_signal = I_filt + 1j * Q_filt
fft_signal = np.fft.fft(complex_signal)

# Compute the power spectral density (PSD)
psd = (1.0 / len(I_filt)) * np.square(np.abs(fft_signal))

# Compute the frequency vector
freq = np.fft.fftfreq(len(I_filt), 1.0/fs)

# Find the index of the peak frequency
max_index = np.argmax(psd)

# Identify the dominant frequency component corresponding to heart rate
heart_rate = freq[max_index]

# Plot the PSD
plt.plot(freq[:len(I_filt)//2], psd[:len(I_filt)//2])
plt.xlim(0, 10)
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density')
plt.title('FFT of Radar Signal')
plt.show()

# Print the estimated heart rate
print('Estimated heart rate:', heart_rate, 'Hz')


