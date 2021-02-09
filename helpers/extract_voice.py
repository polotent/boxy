import numpy as np
from helpers.plotting import plot_energies

def get_energies(frames):
    '''
    Calculates energy for each frame in a frame sequence
    '''
    return np.sum(np.abs(frames), axis=1)

def get_ZCR(frame):
    '''
    Calculates zero crossing rate in a single frame
    '''
    ZCR = 0
    for i in range(len(frame)-2):
        ZCR += 1/2 * np.abs(np.sign(frame[i]) - np.sign(frame[i+1]))
    return ZCR

def get_init_frames(frames, frame_size, hop_size, init_length):
    n_hops = int(np.ceil((init_length - frame_size) / hop_size)) if init_length >= frame_size else 0
    init_frames = frames[:n_hops]
    return init_frames

def get_IZCT(frames, hop_size): 
    ZCR = list()
    for frame in frames:
        ZCR.append(get_ZCR(frame))

    IZC = np.mean(ZCR)
    IZCT = min((25 / hop_size), IZC * 2 * np.std(ZCR))
    return IZCT

def get_IMX(energies):
    return np.max(energies)

def get_IMN(energies):
    return np.min(energies)

def get_I1(IMN, IMX):
    return 0.03 * (IMX - IMN) + IMN

def get_I2(IMN):
    return 4 * IMN

def get_energy_thresholds(I1, I2):
    ITL = min(I1, I2)
    ITU = 10 * ITL
    return ITL, ITU

def get_frame_index_by_energy(energies, ITL, ITU, mode='begin'):
    if mode == 'begin':
        frame_index = 0
        start = 0
        end = len(energies)
        step = 1        
    elif mode == 'end':
        frame_index = len(energies) - 1
        start = len(energies) - 1
        end = -1
        step = -1
    else:
        raise ValueError('get_frame_index_by_energy function \'mode\' argument must be either \'begin\' or \'end\'.')

    cross_ITL_flag = False
    for i in range(start, end, step):
        if energies[i] >= ITL and not cross_ITL_flag:
            cross_ITL_flag = True
            frame_index = i
        if energies[i] < ITL:
            cross_ITL_flag = False
        if energies[i] > ITU:
            return frame_index
    return frame_index

def get_frame_index_by_ZCR(frames, hop_size, suggested_frame_index, search_length, IZCT, mode='begin'):
    search_length_in_frames = int(np.floor(search_length / hop_size))
    frame_index = suggested_frame_index
    start = suggested_frame_index
    if mode == 'begin':
        step = -1
        if frame_index - search_length_in_frames < 0:
            end = -1
        else:
            end = frame_index - search_length_in_frames - 1
    elif mode == 'end':
        step = 1
        if frame_index + search_length_in_frames > len(frames) - 1:
            end = len(frames)
        else:
            end = frame_index + search_length_in_frames + 1
    else:
        raise ValueError('get_frames_by_ZCR function \'mode\' argument must be either \'begin\' or \'end\'.')
    
    IZCT_count = 0
    for i in range(start, end, step):
        if get_ZCR(frames[i]) > IZCT:
            IZCT_count += 1
        if IZCT_count == 3:
            frame_index = i
            return frame_index
    return frame_index

def get_voice_frames(frames, frame_size, hop_size, init_length, search_length):
    init_frames = get_init_frames(frames, frame_size, hop_size, init_length)
    energies = get_energies(frames)

    IMN = get_IMN(energies)
    IMX = get_IMX(energies)

    I1 = get_I1(IMN, IMX)
    I2 = get_I2(IMN)
    ITL, ITU = get_energy_thresholds(I1, I2)
    print('ITL:', ITL, 'ITU:', ITU)
    IZCT = get_IZCT(init_frames, hop_size)

    # development plotting
    # plot_energies(energies, ITL, ITU)
    
    suggested_start_frame_index = get_frame_index_by_energy(energies, ITL, ITU, mode='begin')
    suggested_end_frame_index = get_frame_index_by_energy(energies, ITL, ITU, mode='end')

    start_frame_index = get_frame_index_by_ZCR(frames, hop_size, suggested_start_frame_index, search_length, IZCT, mode='begin')
    end_frame_index = get_frame_index_by_ZCR(frames, hop_size, suggested_end_frame_index, search_length, IZCT, mode='end')

    print('Suggested:', suggested_start_frame_index, suggested_end_frame_index)
    print('Found:', start_frame_index, end_frame_index)

    voice_frames = frames[start_frame_index:end_frame_index]
    return voice_frames