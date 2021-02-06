from scipy.io import wavfile
import helpers as hp

# TODO: make a config file for constants and read about config files

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
    hp.plot_audio(audio, sample_rate)

    audio = hp.normalize_audio(audio)
    audio = hp.remove_DC_offset(audio)
    audio = hp.filter_audio(audio, sample_rate, LOW_PASS, HIGH_PASS)
    frames = hp.split_into_frames(audio, sample_rate, FRAME_SIZE, HOP_SIZE)
    voice_frames = hp.get_voice_frames(frames, FRAME_SIZE, HOP_SIZE, INIT_LENGTH, SEARCH_LENGTH)
    audio = hp.join_frames(voice_frames, sample_rate, HOP_SIZE)

    hp.plot_audio(audio, sample_rate)
    return audio

def process_folder(folder_name):
    # 4-1channel-32bit-float-44100Hz.wav
    sample_rate, audio = wavfile.read('audio/test.wav')
    audio = process_single_audio(audio, sample_rate, '')
    # TODO : save numpy array file
    wavfile.write('audio/extracted_command.wav', sample_rate, audio)


if __name__ == "__main__":
    process_folder("here")    
