import os
from scipy.io import wavfile
from matplotlib.backends.backend_pdf import PdfPages
import helpers as hp
import numpy as np
import logging


# in ms
HOP_SIZE = 5
FRAME_SIZE = 10
INIT_LENGTH = 100
SEARCH_LENGTH = 250

# MFCC
# in frames
UNIFIED_LENGTH = 400
N_COEFFS = 13
N_FILTERS = 26

if HOP_SIZE > FRAME_SIZE:
    raise ValueError('HOP_SIZE constant must be less than or equals FRAME_SIZE')

def process_single_audio(audio, sample_rate, filename):
    original = audio
    hp.print_audio_info(audio, sample_rate, filename)

    audio = hp.normalize_audio(audio)
    audio = hp.remove_DC_offset(audio)
    frames = hp.split_into_frames(audio, sample_rate, FRAME_SIZE, HOP_SIZE)
    voice_frames = hp.get_voice_frames(frames, FRAME_SIZE, HOP_SIZE,
                                       INIT_LENGTH, SEARCH_LENGTH)
    coeffs = hp.mfcc(voice_frames, sample_rate, N_COEFFS, N_FILTERS,
                     low_freq=0, high_freq=(sample_rate // 2))
    extracted = hp.join_frames(voice_frames, sample_rate, HOP_SIZE)
    return original, extracted, coeffs

def process_folder(folder_path, commands_path,
                   generate_report=False, report_path=None):
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    if generate_report:
        if not os.path.isdir(report_path):
            os.mkdir(report_path)
        logging.basicConfig(filename=os.path.join(report_path, 'log.log'), 
                            filemode='w', level=logging.INFO)

    commands, nums = hp.get_commands_dict(commands_path)

    files = list()
    for r, d, f in os.walk(folder_path):
        for file in f:
            if ('.wav' in file):
                rel_dir = os.path.relpath(r, folder_path) if not os.path.relpath(r, folder_path) == '.' else ''
                speaker_name = rel_dir.split(os.sep)[0]
                files.append([os.path.join(r, file), rel_dir, file, speaker_name])

    for i in range(len(files)):
        sample_rate, audio = wavfile.read(files[i][0])
        original, extracted, coeffs = process_single_audio(audio, sample_rate, files[i][0])

        splited = files[i][2].split('-')
        if len(splited) < 2:
            raise Warning('Invalid filename for extracting command from it. Filename must'
                             'contain command after record number. E.g. \'0-command-info.wav\'')
            continue

        command = splited[1]

        if command not in commands:
            raise Warning(f'No command \'{command}\' found in commands dict. Passing over')
            continue
        coeffs = hp.unify_coeffs(coeffs, UNIFIED_LENGTH)

        if generate_report:
            hp.save_compare_audio(original, extracted, sample_rate, files[i][0])

        speaker_name = files[i][3]
        hp.save_to_datafile(os.path.join(save_path, speaker_name + '_data.npy'), np.array(coeffs))
        hp.save_to_datafile(os.path.join(save_path, speaker_name + '_labels.npy'), np.array(command))


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))
    process_folder(folder_path=os.path.join(curr_dir, 'recorded_audio'),
                   commands_path=os.path.join(curr_dir, 'commands.csv'),
                   generate_report=True,
                   report_path=os.path.join(curr_dir, 'logs'))
