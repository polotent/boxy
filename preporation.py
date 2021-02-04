import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import time
from math import floor, ceil
from statistics import stdev

HOP_SIZE = 5
FRAME_SIZE = 10
INIT_LENGTH = 100
SEARCH_LENGTH = 250

if HOP_SIZE > FRAME_SIZE:
    raise ValueError("HOP_SIZE constant must be less than or equals FRAME_SIZE")

def normalize_audio(audio):
    normalized_audio = audio / np.max(np.abs(audio))
    return normalized_audio

def remove_DC_offset(audio):
    balanced_audio = audio - np.mean(audio)
    return balanced_audio

def split_into_frames(audio, sample_rate, frame_size, hop_size):
    hop_size_in_samples = floor(sample_rate * hop_size / 1000)
    frame_size_in_samples = floor(sample_rate * frame_size / 1000)
    n_frames = floor((audio.shape[0] - frame_size_in_samples) / hop_size_in_samples)

    frames = list()
    for i in range(n_frames):
        frames.append(audio[i * hop_size_in_samples : i * hop_size_in_samples + frame_size_in_samples])
    return frames

def join_frames(frames, sample_rate, hop_size):
    hop_size_in_samples = floor(sample_rate * hop_size / 1000)
    audio = list()
    for i in range(len(frames)):
        if i == len(frames) - 1:
            audio = np.concatenate((audio, frames[i]))
        else:
            audio = np.concatenate((audio, frames[i][:hop_size_in_samples]))
    return audio

def calc_energies(frames):
    '''
    Calculates energy for each frame in a frame sequence
    '''
    return np.sum(np.square(frames), axis=1)


def calc_ZCR(frame):
    '''
    Calculates zero crossing rate in a single frame
    '''
    ZCR = 0
    for i in range(len(frame)-2):
        ZCR += 1/2 * np.abs(np.sign(frame[i]) - np.sign(frame[i+1]))
    return ZCR

def get_init_frames(frames, frame_size, hop_size, init_length):
    n_hops = ceil((init_length - frame_size) / hop_size) if init_length >= frame_size else 0
    init_frames = frames[:n_hops]
    return init_frames

def calc_IZCT(frames, hop_size): 
    ZCR = list()
    for frame in frames:
        ZCR.append(calc_ZCR(frame))

    IZC = np.mean(ZCR)
    IZCT = min((25 / hop_size), IZC * 2 * np.std(ZCR))
    return IZCT

def get_IMX(energies):
    return np.max(energies)

def get_IMN(energies):
    return np.min(energies)

def get_I1(IMX, IMN):
    return 0.03 * (IMX - IMN) + IMN

def get_I2(IMN):
    return 4 * IMN

def get_energy_thresholds(I1, I2):
    ITL = min(I1, I2)
    ITU = 5 * ITL
    return ITL, ITU

def get_frame_by_energy(energies, ITL, ITU, mode='begin'):
    if mode == 'begin':
        frame_index = 0
        start = 0
        end = len(energies)
        step = 1        
    elif mode == 'end':
        frame_index = len(energies) - 1
        start = len(energies) - 1
        end = -1
        step = -1
    else:
        raise ValueError("mode argument must be either 'begin' or 'end'.")

    cross_ITL_flag = False
    for i in range(start, end, step):
        if energies[i] > ITL and not cross_ITL_flag:
            cross_ITL_flag = True
            frame_index = i
        if energies[i] < ITL:
            cross_ITL_flag = False
        if energies[i] > ITU:
            return frame_index

    return frame_index

def get_frame_by_ZCR(frames, mode='begin'):
    pass

def get_voice_frames(frames):
    init_frames = get_init_frames(frames, FRAME_SIZE, HOP_SIZE, INIT_LENGTH)
    energies = calc_energies(frames)
    
    IMX = get_IMX(energies)
    IMN = get_IMN(energies)
    I1 = get_I1(IMX, IMN)
    I2 = get_I2(IMN)
    ITL, ITU = get_energy_thresholds(I1, I2)
    IZCT = calc_IZCT(init_frames, HOP_SIZE)
    
    start_frame_index = get_frame_by_energy(energies, ITL, ITU, mode='begin')
    end_frame_index = get_frame_by_energy(energies, ITL, ITU, mode='end')

    voice_frames = frames[start_frame_index:end_frame_index]
    return voice_frames

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
    print_audio_info(audio, sample_rate, filename)
    plot_audio(audio, sample_rate)

    audio = normalize_audio(audio)
    audio = remove_DC_offset(audio)
    frames = split_into_frames(audio, sample_rate, FRAME_SIZE, HOP_SIZE)
    voice_frames = get_voice_frames(frames)
    audio = join_frames(voice_frames, sample_rate, HOP_SIZE)

    plot_audio(audio, sample_rate)
    return audio

def process_folder(folder_name):
    sample_rate, audio = wavfile.read('audio/test1-1channel-32bit-float-44100Hz.wav')
    audio = process_single_audio(audio, sample_rate, 'test-1channel-32bit-float.wav')
    # TODO : save numpy array file
    wavfile.write('audio/extracted_command.wav', sample_rate, audio)

if __name__ == "__main__":
    process_folder("here")    
