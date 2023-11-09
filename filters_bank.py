from scipy import signal
import numpy as np

def create_octave_filter(f0, fs):
    """
    Create an octave bandpass filter.

    Parameters:
        - f0 (float): Center frequency of the filter in Hz.
        - fs (int): Sampling rate.

    Returns:
        - pol_coef (list of ndarrays): Filter coefficients [b, a].

    """
    f1 = octave_rel**(-1/2)*f0
    f2 = octave_rel**(1/2)*f0

    fc1 = f1/(fs*0.5)
    fc2 = f2/(fs*0.5)
    b, a = signal.butter(2, [fc1, fc2], btype='bandpass')
    pol_coef = [b, a]
    return pol_coef

def create_third_octave_filter(f0, fs):
    """
    Create a third-octave bandpass filter.

    Parameters:
        - f0 (float): Center frequency of the filter in Hz.
        - fs (int): Sampling rate.

    Returns:
        - pol_coef (list of ndarrays): Filter coefficients [b, a].

    """
    f1 = (octave_rel**(-1/6))*f0
    f2 = (octave_rel**(1/6))*f0

    fc1 = f1/(fs*0.5)
    fc2 = f2/(fs*0.5)
    b, a = signal.butter(2, [fc1, fc2], btype='bandpass')
    pol_coef = [b, a]
    return pol_coef

def filter_audio(pol_coef, audio):
    """
    Generates a filter for a sinesweep response
    Input:
        - pol_coef: list type object. Numerator (b) and denominator (a) polynomials of the IIR filter. 
        - audio. Array type object.
    Output:
        - filtered_audio: array type object.
    """
    b, a = pol_coef
    filtered_audio = signal.lfilter(b, a, audio)
    return filtered_audio

def sinesweep_filter(f01, f02, fs):
    """
    Generates a filter for a sinesweep response
    Input:
        - f01: first third octave band  
        - f02: last third octave band
        - fs: sample frequency
    Output:
        - pol_coef: list type object. Numerator (b) and denominator (a) polynomials of the IIR filter. 
    """
    #calcula frecuencias laterales del filtro
    f1 = octave_rel**(-1/6)*f01
    f2 = octave_rel**(1/6)*f02

    #normaliza frecuencias
    fc1 = f1/(fs*0.5)
    fc2 = f2/(fs*0.5)

    #genera el filtro
    b, a = signal.butter(1, [fc1, fc2], btype='bandpass')
    pol_coef = [b, a]
    return pol_coef

octave_rel = 2
ref_freq = 1000

