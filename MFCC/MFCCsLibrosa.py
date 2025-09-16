import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

#Load audio
filename = "my_recording.wav"
y, sr = librosa.load(filename, sr=None)

#Parameters
hop_length_seconds = 0.01  # 10 ms hop length
hop_length_samples = int(sr * hop_length_seconds)
n_mfcc = 13  # number of MFCCs 

#Compute MFCCs 
mfccs = librosa.feature.mfcc(
    y=y,
    sr=sr,
    n_mfcc=n_mfcc+1,      # compute 0th coefficient too
    hop_length=hop_length_samples,
    htk=True              # HTK Mel scale
)

#Remove 0th coefficient
mfccs_no0 = mfccs[1:n_mfcc+1, :]  # shape: (13, frames)

#Create time axis (in seconds)
frames = mfccs_no0.shape[1]
time_axis = np.arange(frames) * hop_length_seconds
coeff_axis = np.arange(1, n_mfcc+1)

#Plot heatmap
plt.figure(figsize=(12,6))
plt.pcolormesh(time_axis, coeff_axis, mfccs_no0, shading='auto', cmap='viridis')
plt.xlabel('Time [s]')
plt.ylabel('MFCC Coefficient')
plt.title('MFCC Heatmap (Librosa)')
plt.colorbar(label='Magnitude')
plt.show()
