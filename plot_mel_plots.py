import numpy as np
import helpers as hp


if __name__ == '__main__':
    x = np.linspace(0, 22050, 22050)
    hp.plot_mel_hz(x, hp.hz_to_mel(x))
    hp.plot_mel_filters(hp.get_filters(13,1024, 44100, 0, 22050), 0, 22050)