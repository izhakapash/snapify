import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import butter, filtfilt


def phase_analysis(phase,output):
    plt.plot(phase)
    plt.xlabel('Time (s)')
    plt.ylabel('phase')
    plt.title('phase analysis')
    # Emphasizing less noisy signals
    plt.axvspan(55000,72000, color='red', alpha=0.5)
    plt.axvspan(7000,25000, color='red', alpha=0.5)
    plt.savefig(output + '/phase_plot.png')
    plt.show()


def psd_estimation_plot(freq,psd,output,plot_name):
    # Plot the PSD
    plt.plot(freq, psd)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power spectral density (V^2/Hz)')
    plt.title('PSD of signal')
    plt.xlim(0, 5)
    plt.savefig(output + '/' + plot_name + '_PSD_plot.png')
    plt.show()
    max_index = np.argmax(psd)
    peak = freq[max_index]
    return peak


def BP_filter_signals(signal,fs,lowcut,highcut):
    # Define the filter parameters
    order = 4

    # Compute the filter coefficients
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to the signal
    filtered_signal = filtfilt(b, a, signal)
    return filtered_signal

def run_radar(radar_file,output):
    # uploading the file
    phase = np.load(radar_file)
    # determine sample rate
    fs = 500
    # plot the signal for analysis
    phase_analysis(phase,output)
    # find the less noisy signals on the data
    less_noisy_signal = phase[55000:72000]
    less_noisy_signal_2 = phase[7000:25000]

    # Estimate the PSD using the periodogram function
    plot_name = "before filter"
    freq, psd = signal.periodogram(less_noisy_signal, fs=fs)
    f1 = psd_estimation_plot(freq,psd,output,plot_name)
    print(f"The first frequency in BPM terms is: {60*f1}")

    # first filter between 0.8 to 3 Hz
    plot_name = "after first filter"
    signal_after_filter = BP_filter_signals(less_noisy_signal,fs,0.8,3)
    freq, psd = signal.periodogram(signal_after_filter, fs=fs)
    f2 = psd_estimation_plot(freq,psd,output,plot_name)
    print(f"The second frequency in BPM terms is: {60 * f2}")

    # first filter between 0.8 to 2.4 Hz
    plot_name = "after second filter"
    signal_after_filter = BP_filter_signals(less_noisy_signal, fs, 1,2.4)
    freq, psd = signal.periodogram(signal_after_filter, fs=fs)
    f3 = psd_estimation_plot(freq,psd,output,plot_name)
    print(f"The first frequency in BPM terms is: {60 * f3}")



output_folder = "C:/Users/izhak/OneDrive/Desktop/uotput"  # output path
radar_file = 'C:/Users/izhak/OneDrive/Desktop/phase.npy'  # data path

run_radar(radar_file,output_folder)





