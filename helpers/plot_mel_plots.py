import numpy as np
from mfcc import hz_to_mel, get_filters
from plotting import plot_mel_hz, plot_mel_filters


if __name__ == '__main__':
    x = np.linspace(0, 22050, 22050)
    plot_mel_hz(x, hz_to_mel(x))
    plot_mel_filters(get_filters(13,1024, 44100, 0, 22050), 0, 22050)