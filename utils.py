import matplotlib.pyplot as plt
import numpy as np

def plot_sounds(sound1, sound2, sampling_freq=1, delay=0):
    plt.figure()
    plt.plot(np.arange(0, sound1.shape[0] / sampling_freq, 1/sampling_freq), sound1)
    plt.plot(np.arange(0, sound2.shape[0] / sampling_freq, 1/sampling_freq) + delay, sound2)
    plt.xlabel("Time(s)")
    plt.ylabel("Sound value")
    plt.legend(["Sound 1", "Sound 2"])
    plt.show()