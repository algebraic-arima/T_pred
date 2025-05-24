import os
import numpy as np
import pandas as pd
from scipy.io import wavfile


def wav_to_nparray(wav_path):
    sample_rate, audio_data = wavfile.read(wav_path)
    
    if len(audio_data.shape) == 1:
        audio_data = np.column_stack((audio_data, audio_data))
    elif audio_data.shape[1] > 2:
        audio_data = audio_data[:, :2]
    
    return audio_data

s = wav_to_nparray("./datasets/BWV1080_S.wav")
a = wav_to_nparray("./datasets/BWV1080_A.wav")
t = wav_to_nparray("./datasets/BWV1080_T.wav")
b = wav_to_nparray("./datasets/BWV1080_B.wav")
x = np.concatenate((s, a, t, b), axis=1) * 100
timestamps = np.arange(len(x)).reshape(-1, 1)
time_series_with_timestamps = np.hstack((timestamps, x))

df = pd.DataFrame(time_series_with_timestamps[::5,:], columns=['timestamp', 'sl','sr', 'al','ar', 'tl','tr', 'bl','br'])

df.to_csv("./datasets/BACH.csv", index=False)