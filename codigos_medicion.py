#CODIGO PARA BARRIDO LOGARITMICO

import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from IPython.display import Audio 
import scipy as sp
from scipy.signal import chirp
from scipy.io.wavfile import write
import soundfile as sf


def sineSweep(duration, init_f, end_f, fs=48000):
    """
    Generate a logarithmic sine sweep signal.

    Input:
        - duration (float): Duration of the sine sweep in seconds.
        - init_f (float): Initial frequency of the sine sweep in Hz.
        - end_f (float): Final frequency of the sine sweep in Hz.
        - fs (int, optional): Sampling rate. Default is 48000 Hz.

    Output:
        - Sweep (ndarray): Logarithmic sine sweep signal.
        - t (ndarray): Time values corresponding to the sine sweep signal.
    """
    
    #Genero un tiempo
    t = np.linspace(0, duration, duration*fs)
        
    #Generar señal sinesweep logarítmica
    Sweep = 0.5*chirp(t, f0=init_f, f1=end_f, t1=duration, method='logarithmic', phi=-90)
        
    return Sweep , t

#CODIGO PARA RUIDO ROSA

def ruido_rosa(duration, sr):
    """
    Generate pink noise and save it to a WAV file.

    Input:
        - duration (float): Duration of the pink noise signal in seconds.
        - sr (int): Sampling rate.

    Output:
        None
    """

    samples = np.random.normal(0, 1, int(duration * sr))
    samples /= np.max(np.abs(samples))

    nombre_del_archivo = "ruido_rosa.wav"
    sf.write(nombre_del_archivo, samples, sr)

