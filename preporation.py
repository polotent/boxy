import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time
from math import floor, ceil
from statistics import stdev

HOP_SIZE = 5
FRAME_SIZE = 10

def normalize_audio(audio):
    normalized_audio = audio / np.max(np.abs(audio))
    return normalized_audio

def remove_DC_offset(audio):
    balanced_audio = audio - np.mean(audio)
    return balanced_audio

def split_into_frames(audio, sample_rate, frame_size=10, hop_size=5):
    '''
    sample_rate: in Hz
    frame_size: in ms
    hop_size: in ms
    '''

    hop_size_in_samples = floor(sample_rate * hop_size / 1000)
    frame_size_in_samples = floor(sample_rate * frame_size / 1000)
    n_frames = floor(audio.shape[0] / hop_size_in_samples)

    # in case frame_size > hop_size
    audio = np.concatenate((audio, np.zeros(frame_size_in_samples)))

    frames = list()
    for i in range(n_frames):
        frames.append(audio[i * hop_size_in_samples : i * hop_size_in_samples + frame_size_in_samples])
    return frames


def calc_energy(frames):
    return np.sum(np.square(frames), axis=1)


def _calc_ZCR(frame):
    '''
    func for calculating zero crossing rate in a single frame
    '''
    ZCR = 0
    for i in range(len(frame)-2):
        ZCR += 1/2 * np.abs(np.sign(frame[i]) - np.sign(frame[i+1]))
    return ZCR


def calc_IZCT(frames, hop_size, frame_size, init_length): 
    n_hops = ceil((init_length - frame_size) / hop_size) if init_length >= frame_size else 0
    init_frames = frames[:n_hops]

    ZCR = list()
    for frame in init_frames:
        ZCR.append(calc_zcr(frame))

    IZC = np.mean(ZCR)
    IZCT = min((25 / hop_size), IZC * 2 * np.std(ZCR))
    return IZCT

def print_audio_info(audio, sample_rate, filename):
    print(f'filename: {filename}, sample_rate: {sample_rate}Hz, duration: {(audio.shape[0] / sample_rate):.2f}s, '
          f'chunks: {audio.shape[0]}')

def plot_audio(audio, sample_rate):
    length = audio.shape[0] / sample_rate
    time = np.linspace(0., length, audio.shape[0])
    plt.plot(time, audio)
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.show()


def process_single_audio(audio, sample_rate, filename):
    #print_audio_info(audio, sample_rate, filename)
    #plot_audio(audio, sample_rate)

    audio = normalize_audio(audio)
    audio = remove_DC_offset(audio)
    frames = split_into_frames(audio, sample_rate, FRAME_SIZE, HOP_SIZE)
    energies = calc_energy(frames)
    calc_IZCT()

    return audio

def process_folder(folder_name):
    sample_rate, audio = wavfile.read('audio/test-1channel-32bit-float.wav')
    audio = process_single_audio(audio, sample_rate, 'test-1channel-32bit-float.wav')
    # save to csv or some file sound format dataset (mb numpy array file)


if __name__ == "__main__":
    process_folder("here")    
