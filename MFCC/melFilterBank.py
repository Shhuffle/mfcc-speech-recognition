'''
In this code, triangular mel filter Bank are created. 
Triangular filters are used because they mimic the ear’s critical band response, provide smooth overlapping transitions,
and emphasize the central formant frequencies, which are crucial in speech

'''
import FFT
import numpy as np
import math
import matplotlib.pyplot as plt
#Filter Bank Parameters
NumberOfFilter = 26
Low_freq = 300
High_freq = (FFT.sampling_rate)/2


#variables used 
'''
M = mel scale, frequency -> mel scale
IM = inverse of mel scale, i.e conversion of mel scale to frequency
FFTbins = fourier transform index k (wk) of each windowed frame
IMElbins = frequency obtained form IM converted to its bin equivalent using formula k = (N/fs) * freqency
melenergy = melenergy value obtained form each filterbank, its a matrix
'''





def MelScale(freq):
    M_f = 2595 * np.log10(1+freq/700)
    return M_f 

def IMelScale(M_f):
    freq = 700 * (pow(10,M_f/2595)-1)
    return freq

M = np.linspace(MelScale(Low_freq),MelScale(High_freq),NumberOfFilter+2)
IM = np.zeros(len(M))
for i in range (len(M)):
    IM[i] = IMelScale(M[i])

FFTbins = FFT.Wfft()
#After FFT we have fourier transform coefficients as bin index(k). It is neccessary to map the mel scale frequencies 
#into the bin index (k) using the formula k = (N/fs) * freq.

def mapfreq_fftbin_index(IM):
    IM_bins = np.zeros(len(IM))
    for i in range (len(IM)):
        IM_bins[i] = math.floor((FFT.frame_size * IM[i]) / FFT.sampling_rate)
    return IM_bins

#equivalent weight calculation of bin, returns weight of ith bin.
def triangular_filer(start,center,end,bin_index):
    if bin_index > start and bin_index < center:
        y = (bin_index - start) / (center - start)  
        
    elif bin_index == center:
        y = 1
    elif bin_index > center and bin_index <end:
        y = (end-bin_index)/(end - center) 
    else:
        y = 0
    return y

IMel_bins = mapfreq_fftbin_index(IM)


#mel scale energy calculation, return the log of melenergy list one form each triangular filter
def logmelspec(frame_number,IMel_binsA=IMel_bins,fft_binsA=FFTbins):
    melenergy = np.zeros(NumberOfFilter)
    for i in range(NumberOfFilter):
        filtered_energy = 0
        start = IMel_binsA[i]
        center = IMel_binsA[i+1]
        end = IMel_binsA[i+2]

        for k in range (int(start),int(end)+1):
            weights = triangular_filer(start,center,end,k)
            if k < (FFT.frame_size//2):
                filtered_energy += weights * (fft_binsA[frame_number,k] ** 2)
            
        melenergy[i] = filtered_energy
    return 20*(np.log10(melenergy + 1e-8)) #1e-8 is to avoid log of 0


if __name__ == "__main__":
    frame_numbers = 10
    logmag = logmelspec(frame_numbers,IMel_bins,FFTbins)
    
    plt.stem(np.arange(NumberOfFilter),logmag)
    plt.title(f"Mel Spectrum of frame {frame_numbers}")
    plt.xlabel("Mel Filter Index energy value form each melband triangular filter")
    plt.ylabel("Log Magnitude in dB")
    plt.grid(True)
    plt.show()
    
   
    #plots

