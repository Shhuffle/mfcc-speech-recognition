# MFCCs & Speech Recognition From Scratch

This project demonstrates **MFCC (Mel-Frequency Cepstral Coefficients) calculation from scratch** and uses a **K-Nearest Neighbors (KNN) algorithm** to recognize spoken letters or words.

---

## Required Libraries

Make sure you have the following Python libraries installed:

- `soundfile`
- `numpy`
- `matplotlib`
- `math`

---

## File Descriptions

### 1. `FFT.py`
This is the base code of the program. It performs:

- Takes the input discrete signal from the `.wav` file named `my_recording.wav`.
- Applies **pre-emphasis** to the signal.
- Frames the signal and calculates hop size.
- Applies a **Hanning window** on each frame.
- Performs **FFT** on each windowed frame.

### 2. `melFilterBank.py`
Based on `FFT.py`, this file performs:

- Conversion from Hz to Mel scale.
- Taking natural log on Mel scale.
- Converting log Mel scale back to Hz.
- Converting frequency to FFT bins.
- Implements **triangular filter bank** using the converted FFT bins (26 filters total).
- Calculates Mel scale energy for each filter.

### 3. `DCT.py`
Final step for MFCC calculation:

- Applies **Discrete Cosine Transform (DCT)** on each frame's Mel scale energy (26 values).
- Only the first 13 values (excluding the 0th) are used.
- Computes the **feature vector** by taking the mean of each coefficient across all frames.

### 4. `KNNForMFCCs.py`
A KNN algorithm customized for MFCC-based speech recognition:

- Takes the calculated feature vector of `my_recording.wav` as input.
- `C` is a 2D vector storing known MFCCs, with the first element of each row being the corresponding letter/word.
- `computeDistance` calculates the distance between the unknown input MFCCs and the known MFCCs in `C`.
- Returns the label if the nearest MFCC is within a threshold, otherwise returns `unknown`.

---

## Running the Program

1. Record your sound using `RecordYourVoice.py`. **Do not alter the recording length** if you want to use the provided MFCC values.
2. Run `KNNForMFCCs.py` to recognize the letter/word spoken in `my_recording.wav`.
3. Run `DCT.py` to plot the **heat map of the MFCCs**.
4. Run `FFT.py` to plot the **time and frequency domain representation** of the windowed frame.
5. Run `melFilterBank.py` to plot the **MFCCs of a frame**.

---

## Notes

- The known MFCC values are calculated using the my voice and recorded in a sound environment with no background noise. MFCCs vary with the environment, so for better results, **record your own audio**, calculate MFCCs, and add them to the vector `C` with the first element as the letter/word.
- Known MFCCs are recorded for a **1-second audio**, with frames starting after 0.5 seconds. Altering the audio length will change the MFCC values, so keep the audio 1 second if you want to use the provided values. Otherwise, calculate MFCCs yourself.
- Once you record your own audio and add the MFCCs to the known database value then you can use the program to recognize the letter/word in the next run.
