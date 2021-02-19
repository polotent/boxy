import numpy as np
import matplotlib.pyplot as plt


def save_compare_audio(pdf_report, audio1, audio2, sample_rate, filename):
    fig, (ax1, ax2) = plt.subplots(1,2)

    fig.suptitle(filename, fontsize=8)

    length1 = audio1.shape[0] / sample_rate
    time1 = np.linspace(0., length1, audio1.shape[0])

    length2 = audio2.shape[0] / sample_rate
    time2 = np.linspace(0., length2, audio2.shape[0])
    
    ax1.plot(time1, audio1)
    # ax1.set_xlabel("Time [s]")
    # ax1.set_ylabel("Amplitude")

    ax2.plot(time2, audio2)
    # ax2.set_xlabel("Time [s]")
    # ax2.set_ylabel("Amplitude")

    ax1.set_title(f'Original audio: {length1:.2f}sec', fontsize=8)
    ax2.set_title(f'Extractred voice: {length2:.2f}sec', fontsize=8)

    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    ax2.get_xaxis().set_visible(False)
    ax2.get_yaxis().set_visible(False)

    pdf_report.savefig(fig)
    

    plt.close()
    # plt.show()