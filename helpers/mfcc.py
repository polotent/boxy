import numpy as np
from scipy.fftpack import dct

def hz_to_mel(hz):
    return 1127 * np.log(1 + hz / 700)

def mel_to_hz(mel):
    return 700 * (np.exp(mel / 1127) - 1)

def get_nfft(frame_size):
    i = 1
    while frame_size > i:
        i *= 2
    return i

def get_magnitude_spectrum(frames, nfft):
    return np.absolute(np.fft.rfft(frames, nfft))

def get_power_spectrum(frames, nfft):
    return np.square(get_magnitude_spectrum(frames, nfft))

def get_filters(n_filters, nfft, sample_rate, low_freq, high_freq):
    low_mel = hz_to_mel(low_freq)
    high_mel = hz_to_mel(high_freq)
    mel_points = np.linspace(low_mel, high_mel, n_filters+2)
    spectrum_bin = np.floor((nfft+1) / sample_rate * mel_to_hz(mel_points))
    filters = np.zeros([n_filters, nfft//2+1])
    for j in range(0, n_filters):
        for i in range(int(spectrum_bin[j]), int(spectrum_bin[j+1])):
            filters[j,i] = (i - spectrum_bin[j]) / (spectrum_bin[j+1]-spectrum_bin[j])
        for i in range(int(spectrum_bin[j+1]), int(spectrum_bin[j+2])):
            filters[j,i] = (spectrum_bin[j+2]-i) / (spectrum_bin[j+2]-spectrum_bin[j+1])
    return filters

def get_mel_spectrum(spectrum, filters):
    product = np.dot(spectrum, filters.T)
    product = np.where(product == 0, np.finfo(float).eps, product) # if feat is zero, we get problems with log
    return product

def mfcc(frames, sample_rate, n_coeffs=13, n_filters=26, low_freq=0, high_freq=None): 
    high_freq = high_freq or sample_rate // 2
    nfft = get_nfft(np.shape(frames)[1])
    power_spectrum = get_power_spectrum(frames, nfft)
    filters = get_filters(n_filters, nfft, sample_rate, low_freq, high_freq)
    mel_spectrum = get_mel_spectrum(power_spectrum, filters)
    mel_spectrum = np.log(mel_spectrum)
    cepstral_coeffs = dct(mel_spectrum, type=2, axis=1, norm='ortho')[:,:n_coeffs]
    return np.array(cepstral_coeffs)
