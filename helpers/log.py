import numpy as np
import matplotlib.pyplot as plt
import logging


def print_audio_info(audio, sample_rate, filename):
    print(f'filename: {filename}, sample_rate: {sample_rate}Hz, duration: {(audio.shape[0] / sample_rate):.2f}sec, '
          f'chunks: {audio.shape[0]}')

def save_compare_audio(audio1, audio2, sample_rate, filename):
    length1 = audio1.shape[0] / sample_rate
    time1 = np.linspace(0., length1, audio1.shape[0])

    length2 = audio2.shape[0] / sample_rate
    time2 = np.linspace(0., length2, audio2.shape[0])
    
    logging.info(f'{filename} - [{length1:0.2f}sec -> {length2:0.2f}sec]')
