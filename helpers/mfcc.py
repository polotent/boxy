import numpy as np

def freq_to_mel(freq):
    return 1127 * np.log(1 + freq / 700)

def mel_to_freq(mel):
    return 700 * (np.exp(mel / 1127) - 1)

def mfcc(frames, n_mel_coeffs): 
    pass