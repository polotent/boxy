import numpy as np
import matplotlib.pyplot as plt


def save_compare_audio(report, audio1, audio2, sample_rate, filename):
    length1 = audio1.shape[0] / sample_rate
    time1 = np.linspace(0., length1, audio1.shape[0])

    length2 = audio2.shape[0] / sample_rate
    time2 = np.linspace(0., length2, audio2.shape[0])
    
    report.write(f'{filename}\noriginal length: {length1:4.2f}\nextracted length: {length2:4.2f}\n')
