import matplotlib.pyplot as plt
import numpy as np
import scipy


def load_sound_mono(file_name: str):
    sampling_freq, sound = scipy.io.wavfile.read(file_name)
    
    if len(sound.shape) == 2:
        sound = sound[:, 0] # Get single channel audio

    return sampling_freq, sound


def plot_sounds(sound1, sound2, sampling_freq=1, delay=0):
    plt.figure()
    plt.plot(np.arange(0, sound1.shape[0] / sampling_freq, 1/sampling_freq), sound1)
    plt.plot(np.arange(0, sound2.shape[0] / sampling_freq, 1/sampling_freq) + delay, sound2)
    plt.xlabel("Time(s)")
    plt.ylabel("Sound value")
    plt.legend(["Sound 1", "Sound 2"])
    plt.show()


def get_cross_correlation_spectrogram(sound1:np.ndarray, sound2:np.ndarray, sampling_freq:float, time_resolution:float=0.01, freq_resolution:float=2):
    # Compute spectrograms for sounds
    sample_resolution : int = int(time_resolution * sampling_freq)

    freq_bins : float = sampling_freq / freq_resolution / 2
    nperseg : int = int(2 ** (np.round(np.log2(freq_bins)) + 1)) # Find nearest power of 2 and multiply by 2
    noverlap : int = nperseg - sample_resolution

    f1, t1, Sxx1 = scipy.signal.spectrogram(sound1, nperseg=nperseg, noverlap=noverlap, fs=sampling_freq)
    f2, t2, Sxx2 = scipy.signal.spectrogram(sound2, nperseg=nperseg, noverlap=noverlap, fs=sampling_freq)

    # Zero pad the spectrograms
    Sxx1_padded = np.pad(Sxx1, pad_width=((0,0), (0, Sxx2.shape[1] - 1)))
    Sxx2_padded = np.pad(Sxx2, pad_width=((0,0), (0, Sxx1.shape[1] - 1)))

    # Take FFTs
    fft1 = scipy.fft.rfft(Sxx1_padded, axis=1)
    fft2 = scipy.fft.rfft(Sxx2_padded, axis=1)

    # Multiply together in frequency domain
    fft_product = np.multiply(fft1, np.conjugate(fft2))

    # Take IFFT
    cross_correlation_spectrum = scipy.fft.irfft(fft_product, axis=1)

    # Flatten vertically
    cross_correlation = np.sum(cross_correlation_spectrum, axis=0)

    # Get time scale factor
    effective_sampling_freq = sampling_freq / (sound1.shape[0] + sound2.shape[0] - 1) * (Sxx1.shape[1] + Sxx2.shape[1] - 1)

    # Normalise cross correlation by total mass
    # cross_correlation /= np.sum(cross_correlation) # Seems to break everything... check using interval_correlation_test

    # Normalise cross correlation by length
    cross_correlation /= (cross_correlation.shape[0] / effective_sampling_freq)

    return effective_sampling_freq, cross_correlation


def get_delay(cross_correlation, effective_sampling_freq):
    position = np.argmax(cross_correlation)
    value = cross_correlation[position]

    position = position if position < cross_correlation.shape[0] / 2 else position - cross_correlation.shape[0]
    delay: float = - position / effective_sampling_freq

    return delay, value


if __name__ == "__main__":
    # Load audio files
    sampling_freq, sound1 = scipy.io.wavfile.read("audio/Piano C2.wav")
    sampling_freq, sound2 = scipy.io.wavfile.read("audio/Piano C2 - 2.wav")

    # Compute cross correlation
    effective_sampling_freq, cross_correlation = get_cross_correlation_spectrogram(sound1, sound2, sampling_freq)

    # Get delay
    delay, value = get_delay(cross_correlation, effective_sampling_freq)

    print(f"Delay is {delay}s, with a cross correlation score of {value}")
