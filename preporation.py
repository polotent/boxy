from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
import time
import math


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

    hop_size_in_samples = math.floor(sample_rate * hop_size / 1000)
    frame_size_in_samples = math.floor(sample_rate * frame_size / 1000)
    n_frames = math.floor(audio.shape[0] / hop_size_in_samples)

    # in case frame_size > hop_size
    audio = np.concatenate((audio, np.zeros(frame_size_in_samples)))

    frames = list()
    for i in range(n_frames):
        frames.append(audio[i * hop_size_in_samples : i * hop_size_in_samples + frame_size_in_samples])
    return frames


def calc_energy(frames):
    return np.sum(np.square(frames), axis=1)


def calc_zcr(frames):
    '''
    func for calculating zero crossing rate
    '''

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
    frames = split_into_frames(audio, sample_rate)
    energies = calc_energy(frames)


    return audio

def process_folder(folder_name):
    sample_rate, audio = wavfile.read('audio/test-1channel-32bit-float.wav')
    audio = process_single_audio(audio, sample_rate, 'test-1channel-32bit-float.wav')
    # save to csv or some file sound format dataset (mb numpy array file)


if __name__ == "__main__":
    process_folder("here")    
