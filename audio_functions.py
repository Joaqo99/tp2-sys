import soundfile as sf
from IPython.display import Audio
import numpy as np
import filters_bank as fb

def load_audio(file_name):
    """
    Loads a mono or stereo audio file in audios folder.
    Input:
        - file_name: String type object. The file must be an audio file.
    Output:
        - audio: array type object.
        - fs: sample frequency
        - prints if audio is mono or stereo.
    """
    if type(file_name) != str:
        raise Exception("file_name must be a string")

    audio, fs = sf.read(f"../audios/{file_name}")

#    if audio.shape[1] != 1 and audio.shape[1] != 2:
#        raise Exception("Not valid input")
#
#    if audio.shape[1] == 1:
#        print("El audio es mono")
#
#    if audio.shape[1] == 2:
#        print("El audio es estereo")

    return audio , fs

def play_mono_audio(audio, fs):
    """
    Plays a mono audio
    Inputs:
        - audio: array type object. Audio to play. Must be mono.
        - fs: int type object. Sample rate
    """
    #error handling
    if type(fs) != int:
        raise ValueError("fs must be int")
    
    assert len(audio.shape) == 1, "Audio must be mono"


    return Audio(audio, rate=fs)

def to_mono(audio):
    """
    Converts a stereo audio vector to mono.
    Insert:
        - audio: array type object of 2 rows. Audio to convert.
    Output:
        - audio_mono: audio converted
    """
    #error handling
    if  type(audio) != np.ndarray:
        raise ValueError("audio must be a ndarray")
    if len(audio.shape) == 1:
        raise Exception("Audio is already mono")
    elif audio.shape[0] != 2 and audio.shape[1] != 2: 
        raise Exception("Non valid vector")
    
    #features
    audio_mono = (audio[:,0]/2)+(audio[:,1]/2)
    return audio_mono

def get_audio_time_array(audio, fs):
    """
    Returns audio time array
    Input:
        - audio: array type object.
        - fs: Int type object. Sample rate.
    Output:
        - duration: int type object. Audio duration
        - time_array: array type object.
    """
    #error handling
    if  type(audio) != np.ndarray:
        raise ValueError("audio must be a ndarray")
    if type(fs) != int:
        raise ValueError("fs must be int")
    
    #features
    duration = audio.size // fs
    time_array = np.linspace(0, duration, audio.size)

    return duration, time_array

def scale_amplitude(audio, ref):
    """
    Returns an audio amplitude array in dB scale
    Input:
        - audio: array type object.
        - ref: array type object. Reference audio
    Output:
        - audio_pascal: array type object. Audio in pascal
        - audio_dbSPL: array type object. Audio in dB SPL
    """
    if  type(audio) != np.ndarray:
        raise ValueError("audio must be a ndarray")
    
    ref_rms = np.max(ref)/np.sqrt(2)

    audio_pascal = audio/ref_rms

    eps = np.finfo(float).eps

    audio_dbSPL = 10*np.log10((audio_pascal/0.00002)**2 + eps)
    return audio_pascal, audio_dbSPL

def Leq(signal, ref):
    """
    Calculate the equivalent sound pressure level of a signal

    Parameters
    ----------
    signal : numpy.ndarray
        The audio signal array in Pascal values
    ref : numpy.ndarray
        A reference audio signal array for convert the signal in Pascal if needed

    Returns
    -------
    leq: numpy.ndarray
        The equivalent sound pressure level of the signal
    """
    signal_pascal = scale_amplitude(signal, ref)[0]

    p_r = (2.0e-5)**2.
    signal2 = signal_pascal**2

    leq = 10*np.log10(np.mean(signal2)/(p_r))

    return leq

def leq_by_bands(audio, filter_bank, ref):
    """
    Returns an array of equivalent continous sound pressure level (LEQ) of an audio by bands
    Input:
        - audio: array type object.
        - filter_bank: list type object. It must contain the filters for the desired frequency bands.
    Output:
        - bands_leq: array type object. leq ordered from lowest to highest band.
    """

    bands_amplitudes = []

    for filt in filter_bank:
        filtered_audio = fb.filter_audio(filt, audio)
        bands_amplitudes.append(filtered_audio)
    
    bands_leq = []

    for band in bands_amplitudes:
        leq = Leq(band, ref=ref)
        bands_leq.append(leq)

    return bands_leq

def sum_bands(values):
    """
    Returns the sum of energy bands
    Input:
        - values: list type object. Contains every band value
    Output:
        - total_value: float type object. Sum of energy bands
    """

    values = np.array(values)
    pressure = 10**(values/10)
    sum_pressure = np.sum(pressure)
    eps = np.finfo(float).eps

    total_value = 10*np.log10(sum_pressure + eps)
    return total_value