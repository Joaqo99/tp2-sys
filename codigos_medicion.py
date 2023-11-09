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
    asdasdd
    """
    
    #Genero un tiempo
    t = np.linspace(0, duration, duration*fs)
    
    #Generar señal sinesweep logarítmica
    Sweep = 0.5*chirp(t, f0=init_f, f1=end_f, t1=duration, method='logarithmic', phi=-90)
    
    return Sweep , t

#CODIGO PARA RUIDO ROSA

def ruido_rosa(duration, sr):

    samples = np.random.normal(0, 1, int(duration * sr))
    samples /= np.max(np.abs(samples))

    nombre_del_archivo = "ruido_rosa.wav"
    sf.write(nombre_del_archivo, samples, sr)

