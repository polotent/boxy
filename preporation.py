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

LOW_PASS = 100
HIGH_PASS = 4000

if HOP_SIZE > FRAME_SIZE:
    raise ValueError("HOP_SIZE constant must be less than or equals FRAME_SIZE")

def normalize_audio(audio):
    normalized_audio = audio / np.max(np.abs(audio))
    return normalized_audio

def remove_DC_offset(audio):
    balanced_audio = audio - np.mean(audio)
    return balanced_audio

def filter_audio(audio, sample_rate, low_pass=100, high_pass=4000):
    freq = np.fft.rfft(audio, len(audio))
    time = len(audio) / sample_rate
    low_index = int(np.floor(low_pass * time))
    high_index = int(np.ceil(high_pass * time))
    freq[:low_index] = 0
    freq[high_index:] = 0
    audio = np.fft.irfft(freq)
    return audio

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

def get_energies(frames):
    '''
    Calculates energy for each frame in a frame sequence
    '''
    return np.sum(np.abs(frames), axis=1)

def get_ZCR(frame):
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

def get_IZCT(frames, hop_size): 
    ZCR = list()
    for frame in frames:
        ZCR.append(get_ZCR(frame))

    IZC = np.mean(ZCR)
    IZCT = min((25 / hop_size), IZC * 2 * np.std(ZCR))
    return IZCT

def get_IMX(energies):
    return np.max(energies)

def get_IMN(energies):
    return np.min(energies)

def get_I1(IMN, IMX):
    return 0.03 * (IMX - IMN) + IMN

def get_I2(IMN):
    return 4 * IMN

def get_energy_thresholds(I1, I2):
    ITL = min(I1, I2)
    ITU = 10 * ITL
    return ITL, ITU

def get_frame_index_by_energy(energies, ITL, ITU, mode='begin'):
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
        raise ValueError('get_frame_index_by_energy function \'mode\' argument must be either \'begin\' or \'end\'.')

    cross_ITL_flag = False
    for i in range(start, end, step):
        if energies[i] >= ITL and not cross_ITL_flag:
            cross_ITL_flag = True
            frame_index = i
        if energies[i] < ITL:
            cross_ITL_flag = False
        if energies[i] > ITU:
            return frame_index

    return frame_index


def get_frame_index_by_ZCR(frames, hop_size, suggested_frame_index, search_length, IZCT, mode='begin'):
    search_length_in_frames = int(np.floor(search_length / hop_size))
    frame_index = suggested_frame_index
    start = suggested_frame_index
    if mode == 'begin':
        step = -1
        if frame_index - search_length_in_frames < 0:
            end = -1
        else:
            end = frame_index - search_length_in_frames - 1
    elif mode == 'end':
        step = 1
        if frame_index + search_length_in_frames > len(frames) - 1:
            end = len(frames)
        else:
            end = frame_index + search_length_in_frames + 1
    else:
        raise ValueError('get_frames_by_ZCR function \'mode\' argument must be either \'begin\' or \'end\'.')
    
    IZCT_count = 0
    for i in range(start, end, step):
        if get_ZCR(frames[i]) > IZCT:
            IZCT_count += 1
        if IZCT_count == 3:
            frame_index = i
            return frame_index
    return frame_index

def get_voice_frames(frames):
    init_frames = get_init_frames(frames, FRAME_SIZE, HOP_SIZE, INIT_LENGTH)
    energies = get_energies(frames)

    IMN = get_IMN(energies)
    IMX = get_IMX(energies)

    I1 = get_I1(IMN, IMX)
    I2 = get_I2(IMN)
    ITL, ITU = get_energy_thresholds(I1, I2)
    print(ITL, ITU)
    IZCT = get_IZCT(init_frames, HOP_SIZE)

    # development plotting
    plot_energies(energies, ITL, ITU)
    
    suggested_start_frame_index = get_frame_index_by_energy(energies, ITL, ITU, mode='begin')
    suggested_end_frame_index = get_frame_index_by_energy(energies, ITL, ITU, mode='end')

    start_frame_index = get_frame_index_by_ZCR(frames, HOP_SIZE, suggested_start_frame_index, SEARCH_LENGTH, IZCT, mode='begin')
    end_frame_index = get_frame_index_by_ZCR(frames, HOP_SIZE, suggested_end_frame_index, SEARCH_LENGTH, IZCT, mode='end')

    print('Suggested:', suggested_start_frame_index, suggested_end_frame_index)
    print('Found:', start_frame_index, end_frame_index)

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

def plot_energies(energies, ITL, ITU):
    x_linspace = np.linspace(0., len(energies), len(energies))
    ITL_points = [ITL for i in range(len(energies))]
    ITU_points = [ITU for i in range(len(energies))]

    plt.plot(x_linspace, energies, label='Energy')
    plt.plot(x_linspace, ITL_points, label='ITL')
    plt.plot(x_linspace, ITU_points, label='ITU')

    plt.xlabel('frame')
    plt.ylabel('sum(|x(n)|) i=0 ... (length(frame)-1)')
    plt.legend() 
    plt.show()

def process_single_audio(audio, sample_rate, filename):
    print_audio_info(audio, sample_rate, filename)
    plot_audio(audio, sample_rate)

    audio = normalize_audio(audio)
    audio = remove_DC_offset(audio)
    audio = filter_audio(audio, sample_rate, LOW_PASS, HIGH_PASS)
    frames = split_into_frames(audio, sample_rate, FRAME_SIZE, HOP_SIZE)
    voice_frames = get_voice_frames(frames)
    audio = join_frames(voice_frames, sample_rate, HOP_SIZE)

    plot_audio(audio, sample_rate)
    return audio

def process_folder(folder_name):
    # 4-1channel-32bit-float-44100Hz.wav
    sample_rate, audio = wavfile.read('audio/test.wav')
    audio = process_single_audio(audio, sample_rate, '')
    # TODO : save numpy array file
    wavfile.write('audio/extracted_command.wav', sample_rate, audio)

if __name__ == "__main__":
    process_folder("here")    
