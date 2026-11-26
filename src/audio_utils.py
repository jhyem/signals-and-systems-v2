import soundfile as sf
import numpy as np


def read_wav(path):
    """
    Read a WAV file.

    Returns:
        waveform: numpy array [T] or [T, C]
        sample_rate: int
    """
    waveform, sample_rate = sf.read(path, dtype="float32")
    return waveform, sample_rate


def write_wav(path, waveform, sample_rate):
    """
    Save waveform as WAV file.
    """
    waveform = np.asarray(waveform, dtype=np.float32)
    sf.write(path, waveform, sample_rate)
