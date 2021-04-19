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
    
def plot_metrics(history):
    fig = plt.figure(figsize=(12, 5))

    fig.add_subplot(121)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'])

    fig.add_subplot(122)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'])
    plt.show()

def plot_confusion_matrix(df_cm_arr):
    fig, ax = plt.subplots(nrows=len(df_cm_arr) // 2, ncols=2, figsize=(20,10))
    for i, df_cm in enumerate(df_cm_arr):
        cm = np.diag(np.diag(df_cm['cm'].corr()))
        cm = np.delete(cm, (-1), axis=0)
        cbar = (i == len(df_cm_arr))
        sn.heatmap(df_cm['cm'], annot=True, vmin=0.0, vmax=1.0, cmap='gray_r', cbar=cbar, linewidths=2, linecolor='black', mask=cm, ax=ax[i//2,i%2])
        ax[i//2,i%2].set_title(f'Threshold: {df_cm["threshold"]}')
    
    # sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=1.0, cmap='gray_r', linewidths=2, linecolor='black', mask=cm)
        ax[i//2,i%2].set_ylabel('predicted class')
        ax[i//2,i%2].set_xlabel('actual class')
    plt.show()