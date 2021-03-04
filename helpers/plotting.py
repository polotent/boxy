import numpy as np
import matplotlib.pyplot as plt

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
