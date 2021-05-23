import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

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
    
def plot_metrics(history, save_path=None):
    fig = plt.figure(figsize=(12, 5))

    fig.add_subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Функция потерь vs. эпохи обучения')
    plt.ylabel('Значения функции потерь')
    plt.xlabel('Эпохи')
    plt.legend(['Тренировочные данные', 'Валидационные данные'])

    fig.add_subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Функция точности vs. эпохи обучения')
    plt.ylabel('Значения функции точности')
    plt.xlabel('Эпохи')
    plt.legend(['Тренировочные данные', 'Валидационные данные'])
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix(df_cm_arr, save_path=None):
    fig, ax = plt.subplots(nrows=len(df_cm_arr) // 2, ncols=2, figsize=(17,10))
    plt.subplots_adjust(wspace=0.1, hspace=0.45)
    for i, df_cm in enumerate(df_cm_arr):
        cm = np.diag(np.diag(df_cm['cm'].corr()))
        cm = np.delete(cm, (-1), axis=0)
        sn.heatmap(df_cm['cm'], annot=True, vmin=0.0, vmax=1.0, cmap='gray_r', cbar=True, linewidths=2, linecolor='black', mask=cm, ax=ax[i//2,i%2])
        ax[i//2,i%2].set_title(f'Пороговое значение: {df_cm["threshold"]}')
        ax[i//2,i%2].set_ylabel('Распознанная команда')
        ax[i//2,i%2].set_xlabel('Настоящая команда')
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_mel_hz(x, f):
    plt.rcParams.update({'font.size': 22})
    plt.plot(x, f)
    plt.xlim(xmin=0, xmax=22050)
    plt.ylim(ymin=0, ymax=4000)
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.xticks([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
                11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000],
               [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
    plt.xlabel('Частота, кГц')
    plt.ylabel('Частота, мел')
    plt.show()

def plot_mel_filters(filters, low_hz, high_hz):
    plt.rcParams.update({'font.size': 22})
    for filter in filters:
        plt.plot(np.linspace(low_hz, high_hz, filters.shape[1]), filter)
    plt.xlim(xmin=0)
    plt.ylim(ymin=0)
    plt.grid(color='grey', linestyle='dashed', linewidth=1)
    plt.xlabel('Частота, Гц')
    plt.ylabel('Значения функции H')
    plt.show()
