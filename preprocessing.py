import os
from scipy.io import wavfile
from python_speech_features import mfcc
import helpers as hp
import numpy as np


# in ms
HOP_SIZE = 5
FRAME_SIZE = 10
INIT_LENGTH = 100
SEARCH_LENGTH = 250

# in Hz
LOW_PASS = 100
HIGH_PASS = 4000

if HOP_SIZE > FRAME_SIZE:
    raise ValueError("HOP_SIZE constant must be less than or equals FRAME_SIZE")

def process_single_audio(audio, sample_rate, filename):
    hp.print_audio_info(audio, sample_rate, filename)
    # hp.plot_audio(audio, sample_rate)

    audio = hp.normalize_audio(audio)
    audio = hp.remove_DC_offset(audio)
    frames = hp.split_into_frames(audio, sample_rate, FRAME_SIZE, HOP_SIZE)
    voice_frames = hp.get_voice_frames(frames, FRAME_SIZE, HOP_SIZE, INIT_LENGTH, SEARCH_LENGTH)
    # coeffs = hp.mfcc(voice_frames, sample_rate, n_coeffs=13, n_filters=26, low_freq=0, high_freq=(sample_rate // 2))    
    audio = hp.join_frames(voice_frames, sample_rate, HOP_SIZE)
    # hp.plot_audio(audio, sample_rate)
    ### --- python_speech_features version
    coeffs = mfcc(audio, sample_rate, winlen=FRAME_SIZE/1000, winstep=HOP_SIZE/1000) 
    ### ---
    return audio, coeffs

def process_folder(folder_path, save_path):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    files = list()
    for r, d, f in os.walk(folder_path):
        for file in f:
            if ('.wav' in file):
                rel_dir = os.path.relpath(r, folder_path) if not os.path.relpath(r, folder_path) == '.' else ''
                files.append([os.path.join(r, file), rel_dir])

    for i in range(len(files)):
        try:        
            sample_rate, audio = wavfile.read(files[i][0])
            audio, coeffs = process_single_audio(audio, sample_rate, files[i][0])
            
            # path = os.path.join(save_path, files[i], '_extracted.npy')
            # np.save(path, coeffs)

            basename = os.path.basename(files[i][0])
            path = os.path.join(save_path, files[i][1], f'{os.path.splitext(basename)[0]}_extracted.wav')
            if not os.path.isdir(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))

            wavfile.write(path, sample_rate, audio)
        except Exception as e:
            print(e) 


if __name__ == '__main__':
    process_folder('D:\\MyFiles\\projects\\boxy\\recorded_audio',
                   'D:\\MyFiles\\projects\\boxy\\data')    
