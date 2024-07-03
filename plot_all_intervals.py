import numpy as np
import scipy
import matplotlib.pyplot as plt
import utils

# Create arrays of note names and corresponding intervals (relative to A0)

scale : list[str] = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

notes : list[str]= []

for i in range(8):
    for note in scale:
        notes.append(f"{note}{i}")

notes : np.ndarray = np.array(notes)
intervals : np.ndarray = np.arange(notes.shape[0])



for constant_note_index in range(notes.shape[0]):
    constant_note_name : str = notes[constant_note_index]

    try:
        # Get cross correlation values
        sampling_freq, sound1 = utils.load_sound_mono(f"samples/{constant_note_name}.wav")

        temp_intervals = intervals - constant_note_index

        delays : np.ndarray = np.zeros_like(notes, dtype=np.float64)
        values : np.ndarray = np.zeros_like(notes, dtype=np.float64)
        lengths : np.ndarray = np.zeros_like(notes, dtype=np.float64)

        for i in range(notes.shape[0]):
            current_note_name : str = notes[i]
            
            try:
                sampling_freq, sound2 = utils.load_sound_mono(f"samples/{current_note_name}.wav")

                effective_sampling_freq, cross_correlation = utils.get_cross_correlation_spectrogram(sound1, sound2, sampling_freq, time_resolution=.1, freq_resolution=1)
                delay, value = utils.get_delay(cross_correlation, effective_sampling_freq)

                delays[i] = delay
                values[i] = value
                lengths[i] = (sound1.shape[0] + sound2.shape[0]) / sampling_freq

            except FileNotFoundError:
                # print(f"No file for note {current_note_name}")
                pass

        plt.figure()
        plt.plot(temp_intervals, values)
        plt.xlabel("Interval (semitones)")
        plt.ylabel("Cross correlation peak value")
        plt.title(f"Inital sound is {constant_note_name} at index {constant_note_index}")
        plt.savefig(f"temp/{constant_note_index}")
        plt.close()
        # plt.show()

        print(f"Finished {constant_note_name} at index {constant_note_index}")
        
    except FileNotFoundError:
        print(f"No file for note {constant_note_name} at index {constant_note_index}")
