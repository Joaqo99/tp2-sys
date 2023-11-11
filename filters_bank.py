from scipy import signal
import numpy as np

def create_octave_filter(f0, fs, order):
    """
    Creates an octave bandpass filter using second order sections.
    Input:
        - f0 (float): Center frequency of the filter in Hz.
        - fs (int): Sampling rate.

    Output:
        - sos (list of ndarrays): Second order sections for the filter.
    """
    f1 = octave_rel**(-1/2)*f0
    f2 = octave_rel**(1/2)*f0

    fc1 = f1/(fs*0.5)
    fc2 = f2/(fs*0.5)
    sos = signal.butter(order, [fc1, fc2], btype='bandpass', output='sos')
    return sos

def create_third_octave_filter(f0, fs, order):
    """
    Creates a third of octave bandpass filter using second order sections.
    Input:
        - f0 (float): Center frequency of the filter in Hz.
        - fs (int): Sampling rate.

    Output:
        - sos (ndarrays): Second order sections for the filter.
    """
    f1 = (octave_rel**(-1/6))*f0
    f2 = (octave_rel**(1/6))*f0

    fc1 = f1/(fs*0.5)
    fc2 = f2/(fs*0.5)
    sos = signal.butter(order, [fc1, fc2], btype='bandpass', output='sos')
    return sos

def filter_audio(sos, audio):
    """
    Returns a filtered auido. The filter must be expressed in a Second order section method.
    Input:
        - sos. 
        - audio. Array type object.
    Output:
        - filtered_audio: array type object.
    """
    filtered_audio = signal.sosfilt(sos, audio)
    return filtered_audio

def sinesweep_filter(f01, f02, fs):
    """
    Generates a filter for a sinesweep response
    Input:
        - f01: first third octave band  
        - f02: last third octave band
        - fs: sample frequency
    Output:
        - sos: array type object. Second order sections of the filter. 
    """
    #calcula frecuencias laterales del filtro
    f1 = octave_rel**(-1/6)*f01
    f2 = octave_rel**(1/6)*f02

    #normaliza frecuencias
    fc1 = f1/(fs*0.5)
    fc2 = f2/(fs*0.5)

    #genera el filtro
    sos = signal.butter(4, [fc1, fc2], btype='bandpass', output='sos')
    return sos

octave_rel = 2
ref_freq = 1000

