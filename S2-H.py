import numpy as np
import matplotlib.pyplot as plt

def gaussian_filter(data, sigma=1):
    kernel_size = int(6 * sigma + 1)
    kernel = np.exp(-np.linspace(-3, 3, kernel_size)**2 / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return np.convolve(data, kernel, mode='same')

def mean_filter(data, size=5):
    kernel = np.ones(size) / size
    return np.convolve(data, kernel, mode='same')

def butterworth_filter(data, cutoff, fs, order=4, filter_type='low'):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    freqs = np.fft.fftfreq(len(data), d=1/fs)
    H = 1 / (1 + (freqs / normal_cutoff)**(2 * order)) if filter_type == 'low' else 1 - (1 / (1 + (freqs / normal_cutoff)**(2 * order)))
    return np.fft.ifft(np.fft.fft(data) * H).real

def median_filter(data, size=5):
    half_size = size // 2
    filtered = np.copy(data)
    for i in range(half_size, len(data) - half_size):
        filtered[i] = np.median(data[i - half_size:i + half_size + 1])
    return filtered

def laplacian_filter(data):
    laplacian_kernel = np.array([1, -2, 1])
    return np.convolve(data, laplacian_kernel, mode='same')

def high_pass_filter(data, cutoff, fs, order=4):
    return butterworth_filter(data, cutoff, fs, order, 'high')

def low_pass_filter(data, cutoff, fs, order=4):
    return butterworth_filter(data, cutoff, fs, order, 'low')


def plot_spectrogram(data, fs, title):
    plt.figure(figsize=(10, 5))
    plt.specgram(data, NFFT=256, Fs=fs, cmap='inferno')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.title(title)
    plt.colorbar(label='Power')
    plt.show()
    
def plot_time_frequency(data, fs, title):
    plt.figure(figsize=(10, 4))
    plt.plot(np.linspace(0, len(data)/fs, len(data)), data)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.show()
    
    freqs = np.fft.fftfreq(len(data), d=1/fs)
    fft_data = np.abs(np.fft.fft(data))
    plt.figure(figsize=(10, 4))
    plt.plot(freqs[:len(freqs)//2], fft_data[:len(fft_data)//2])
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.title(title + " - Frequency Domain")
    plt.show()

# Generate sample signal
fs = 1000  # Sampling frequency
T = 1.0    # Duration in seconds
t = np.linspace(0, T, int(fs * T), endpoint=False)
signal_data = np.sin(2 * np.pi * 50 * t) + np.sin(2 * np.pi * 120 * t) + np.random.normal(0, 0.5, len(t))

# Apply filters
filtered_gaussian = gaussian_filter(signal_data, sigma=5)
filtered_mean = mean_filter(signal_data, size=10)
filtered_median = median_filter(signal_data, size=5)
filtered_laplacian = laplacian_filter(signal_data)
filtered_highpass = high_pass_filter(signal_data, cutoff=50, fs=fs)
filtered_lowpass = low_pass_filter(signal_data, cutoff=50, fs=fs)

# Plot time, frequency, and spectrogram
plot_time_frequency(signal_data, fs, "Original Signal")
plot_spectrogram(signal_data, fs, "Original Signal Spectrogram")

plot_time_frequency(filtered_gaussian, fs, "Gaussian Filter")
plot_spectrogram(filtered_gaussian, fs, "Gaussian Filter Spectrogram")

plot_time_frequency(filtered_mean, fs, "Mean Filter")
plot_spectrogram(filtered_mean, fs, "Mean Filter Spectrogram")

plot_time_frequency(filtered_median, fs, "Median Filter")
plot_spectrogram(filtered_median, fs, "Median Filter Spectrogram")

plot_time_frequency(filtered_laplacian, fs, "Laplacian Filter")
plot_spectrogram(filtered_laplacian, fs, "Laplacian Filter Spectrogram")

plot_time_frequency(filtered_highpass, fs, "High-Pass Filter")
plot_spectrogram(filtered_highpass, fs, "High-Pass Filter Spectrogram")

plot_time_frequency(filtered_lowpass, fs, "Low-Pass Filter")
plot_spectrogram(filtered_lowpass, fs, "Low-Pass Filter Spectrogram")
