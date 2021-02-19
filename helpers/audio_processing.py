import numpy as np


def normalize_audio(audio):
    normalized_audio = audio / np.max(np.abs(audio))
    return normalized_audio

def remove_DC_offset(audio):
    balanced_audio = audio - np.mean(audio)
    return balanced_audio

def split_into_frames(audio, sample_rate, frame_size, hop_size):
    hop_size_in_samples = int(np.floor(sample_rate * hop_size / 1000))
    frame_size_in_samples = int(np.floor(sample_rate * frame_size / 1000))
    n_frames = int(np.floor((audio.shape[0] - frame_size_in_samples) / hop_size_in_samples))

    frames = list()
    for i in range(n_frames):
        frames.append(audio[i * hop_size_in_samples : i * hop_size_in_samples + frame_size_in_samples])
    return frames

def join_frames(frames, sample_rate, hop_size):
    hop_size_in_samples = int(np.floor(sample_rate * hop_size / 1000))
    audio = np.array(list())
    for i in range(len(frames)):
        if i == len(frames) - 1:
            audio = np.concatenate((audio, frames[i]))
        else:
            audio = np.concatenate((audio, frames[i][:hop_size_in_samples]))
    return audio
