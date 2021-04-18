import numpy as np
import matplotlib.pyplot as plt
import logging


def print_audio_info(audio, sample_rate, abs_file_path):
    print(f'{abs_file_path}, sample_rate: {sample_rate}Hz, duration: {(audio.shape[0] / sample_rate):.2f}sec, '
          f'chunks: {audio.shape[0]}')

def save_compare_audio(audio1, audio2, sample_rate, short_file_path):
    length1 = audio1.shape[0] / sample_rate
    time1 = np.linspace(0., length1, audio1.shape[0])

    length2 = audio2.shape[0] / sample_rate
    time2 = np.linspace(0., length2, audio2.shape[0])
    
    logging.info(f'{short_file_path} - [{length1:0.2f}sec -> {length2:0.2f}sec]')

def save_max_len_in_frames(max_len_in_frames):
    logging.info(f'Maximum number of frames : {max_len_in_frames} frames.')

def log_finish():
    logging.info('Finished processing data.')
