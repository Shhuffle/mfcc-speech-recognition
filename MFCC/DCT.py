'''
In this code first we will create a 2D mel spectrum vector where logarithmic energy form each triangular filter for each frame 
will me stored. Then DCT will be applied on each frame keeping the first 13 coefficients ignore the 0th coefficient.
Finally we will print the MFCCs for certain frames.

'''
import melFilterBank
import numpy as np
from scipy.fft import dct
import matplotlib.pyplot as plt


#2D array of melspecturm with energy values form each triangular mel filter for each frame 
melspectrum =  np.zeros((melFilterBank.FFT.total_frames,melFilterBank.NumberOfFilter), dtype=float)

for i in range(melFilterBank.FFT.total_frames):
    melspectrum[i,:] = melFilterBank.logmelspec(i)


def performdct(melspec):
    
    cepstrum = np.zeros((melFilterBank.FFT.total_frames, melFilterBank.NumberOfFilter), dtype=float)
    for i in range(melFilterBank.FFT.total_frames):
        cepstrum[i,:] = dct(melspec[i,:], type=2, norm='ortho')
    return cepstrum[:,1:14]  # keep 2nd to 13th coefficients

MFCCS = performdct(melspectrum)
feature_vector = MFCCS.mean(axis=0)

if __name__ == "__main__":

    #print MFCCS
    
   
    frame_number = 40
    print(f"The UnNormalized MFCCs value for frame {frame_number} is",MFCCS[frame_number,:])
    # print(f"Total frames {melFilterBank.FFT.total_frames}")

    
    mfcc_norm = (MFCCS - MFCCS.mean(axis=0)) / MFCCS.std(axis=0)
    
    print(f"Normalized MFCCs for frame {frame_number}:")
    print(mfcc_norm[frame_number, :])
    print("Featured_Vector: ",feature_vector)

   #plot heat map to visualize       
    MFCCS = performdct(melspectrum)  # shape: (total_frames × 13)
    time_axis = np.arange(mfcc_norm.shape[0]) * melFilterBank.FFT.hop_length 
    coeff_axis = np.arange(1, mfcc_norm.shape[1]+1)

    fig,axs = plt.subplots(1,2,figsize = (12,6))
    pcm = axs[0].pcolormesh(time_axis, coeff_axis, mfcc_norm.T, shading='auto', cmap='viridis') 
    axs[0].pcolormesh(time_axis, coeff_axis, mfcc_norm.T, shading='auto', cmap='viridis')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('MFCC Coefficient')
    axs[0].set_title('Feature MFCCs')
    fig.colorbar(pcm, ax=axs[0], label='Magnitude')

    time_axis = np.arange(MFCCS.shape[0]) * melFilterBank.FFT.hop_length 
    coeff_axis = np.arange(1, MFCCS.shape[1]+1)

    pcm = axs[1].pcolormesh(time_axis, coeff_axis, MFCCS.T, shading='auto', cmap='viridis') 
    axs[1].pcolormesh(time_axis, coeff_axis, MFCCS.T, shading='auto', cmap='viridis')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('MFCC Coefficient')
    axs[1].set_title('MFCC Heatmap without Normalization')
    fig.colorbar(pcm, ax=axs[1], label='Magnitude')
    plt.show()







