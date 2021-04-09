from pathlib import Path
import numpy as np
import os


def save_to_datafile(file_path, data):
    p = Path(file_path)
    with p.open('ab') as f:
        np.save(f, data)

def load_from_datafile(file_path):
    p = Path(file_path)
    with p.open('rb') as f:
        fsz = os.fstat(f.fileno()).st_size
        out = list() 
        out.append(np.load(f))
        while f.tell() < fsz:
            out.append(np.load(f))
    return np.array(out)