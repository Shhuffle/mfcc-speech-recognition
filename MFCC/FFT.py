'''
In this code following operations are performed:
1. Take the input audio using soundfile
2. PreEmphasize the input discrete signal to boost the higher frequencies energy. Human Speech generally contains low energy
    high frequencies signal even though  it contain important formants (resonanes that distinguish vowel and consonants).
    So it is necessary to boost the high frequency signal (1khz-8khz)
3. Windowing of the preEmphasized signal. Hanning function is used in this code. Windowing allows smooth transition between the
    frames preventing frequency leakage.
4. FFt on windowed signal. We generate a frequeny spectrum for each frame using numpy fft library.


'''

import soundfile as sf
import numpy as np 
import matplotlib.pylab as plt


# Load audio
filename = "my_recording.wav"
discrete_input, sampling_rate = sf.read(filename)


def PreEmphasis(x_n,alpha = 0.97):
    y = np.zeros(len(x_n))
    y[0] = x_n[0]
    for n in range(1,len(x_n)):
        y[n] = x_n[n] - alpha * x_n[n-1] #first order high pass filter 
    return y

#emphasized signal 
frame_start = 22000 #The value is set to 22000 because it corresponds to 0.5s, which is the value after which the program
#RecordYourVoice starts to record sound in most cases.
x = PreEmphasis(discrete_input[frame_start:,0])

#Frame parameters
frame_length  = 0.025 #25ms
hop_length = 0.01 #10ms
hop_size = int(sampling_rate * hop_length)
frame_size = int(sampling_rate * frame_length)
total_frames = int((len(x) - frame_size) / hop_size)
print("Total Frames ",total_frames)
def frames(x,start):
    next_frame = start+hop_size
    current_frame_val =  x[start:start+frame_size]
    
    return next_frame, current_frame_val


def Windowing(x,frame_start):
    next_start ,x_f = frames(x,frame_start)
    w = np.hanning(len(x_f))
    windowed_frame = x_f * w
    return next_start, windowed_frame

#The following function will return a 2D list (row = frame, column = Fourier Transform X(ejw)) which contains the 
#DTFT of all the frames. This will be later used to generate mel 
def Wfft(start = 0):
    X = np.zeros((total_frames,frame_size//2),dtype=complex)
    
    for i in range(total_frames):
        start,windowed_frame= Windowing(x,start)

        X[i,:] = np.fft.fft(windowed_frame)[:frame_size//2] #positive frequencies only
       
    return (np.abs(X)) #We are interesed only in the real part 



if __name__ == "__main__":
    xf = Wfft(0)
    fig,axs = plt.subplots(2,2,figsize = (6,6))
    plt.tight_layout(pad=2, h_pad=4, w_pad=3)
    k = np.arange(frame_size//2)    #up to Nyquist frequency
    frequency_axis = (k *sampling_rate) / (frame_size*1000) #in Khz

    # Plot the magnitude spectrum
    frame_number = 20
    axs[1,0].plot(frequency_axis, xf[frame_number,:])
    axs[1,0].set_title(f"Magnitude Spectrum of frame {frame_number}")
    axs[1,0].set_xlabel("Frequency [kHz]")
    axs[1,0].set_ylabel("Magnitude")
    axs[1,0].grid(True)

    frame_number = 21
    axs[1,1].plot(frequency_axis, xf[frame_number,:])  
    axs[1,1].set_title(f"Magnitude Spectrum of frame {frame_number}")
    axs[1,1].set_xlabel("Frequency [kHz]")
    axs[1,1].set_ylabel("Magnitude")
    axs[1,1].grid(True)
    



    frame_number = 20
    _, win_frame200 = Windowing(x, frame_number*hop_size)
    t200 = np.arange(len(win_frame200)) / sampling_rate
    axs[0,0].plot(t200, win_frame200)
    axs[0,0].set_title(f"Windowed Frame {frame_number} (Time Domain)")
    axs[0,0].set_xlabel("Time [s]")
    axs[0,0].set_ylabel("Amplitude")
    axs[0,0].grid(True)

    frame_number = 21
    _, win_frame201 = Windowing(x, frame_number*hop_size)
    t201 = np.arange(len(win_frame201)) / sampling_rate
    axs[0,1].plot(t201, win_frame201)
    axs[0,1].set_title(f"Windowed Frame {frame_number} (Time Domain)")
    axs[0,1].set_xlabel("Time [s]")
    axs[0,1].set_ylabel("Amplitude")
    axs[0,1].grid(True)

    plt.show()


    