import soundfile as sf
from IPython.display import Audio
import numpy as np
import filters_bank as fb
import read_files as rf
from scipy import signal

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

    audio, fs = sf.read(f"{file_name}")

    return audio , fs

def save_audio(file_name, audio, fs=48000):
    """
    Save an audio signal to a file in WAV format.

    Parameters:
        - file_name (str): Name of the output WAV file.
        - audio (ndarray): Audio signal to save.
        - fs (int, optional): Sampling rate. Default is 48000.

    Returns:
        None
    """
    if type(file_name) != str:
        raise Exception("file_name must be a string")

    sf.write(file_name, audio, fs)

    return 

def play_mono_audio(audio, fs):
    """
    Play a mono audio.

    Parameters:
        - audio: array-like
            Mono audio signal to play.
        - fs: int
            Sample rate.

    Returns:
        - Audio object: An object for playing audio.
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
    Returns an audio amplitude array in pascal and dB scales
    Input:
        - audio: array type object.
        - ref: array type object. Calibration audio
    Output:
        - audio_pascal: array type object. Audio in pascal
        - audio_dbSPL: array type object. Audio in dB SPL
    """
    if  type(audio) != np.ndarray or type(ref) != np.ndarray:
        raise ValueError("Both audio and reference must be nd arrays")
    
    ref_rms = np.max(ref)/np.sqrt(2)

    audio_pascal = audio/ref_rms

    eps = np.finfo(float).eps

    audio_dbSPL = 10*np.log10((audio_pascal/0.00002)**2 + eps)
    return audio_pascal, audio_dbSPL

def Leq(signal, ref):
    """
    Calculate the equivalent sound pressure level of a signal
    Input:
        - signal: array type object.
        - ref: array type object. Calibration audio
    Output:
        leq: numpy sacalar array. The equivalent sound pressure level of the signal
    """

    if  type(signal) != np.ndarray or type(ref) != np.ndarray:
        raise ValueError("Both signal and reference must be nd arrays")

    signal_pascal = scale_amplitude(signal, ref)[0]

    p_r = (20.0e-6)**2.
    signal2 = signal_pascal**2

    leq = 10*np.log10(np.mean(signal2)/(p_r))

    return leq

def leq_by_bands(audio, filter_bank, ref):
    """
    Returns an array of equivalent continous sound pressure level (LEQ) of an audio by bands
    Inputs:
        - audio: array type object.
        - filter_bank: list type object. It must contain the filters for the desired frequency bands.
    Output:
        - bands_leq: array type object. leq ordered from lowest to highest band.
    """

    if  type(audio) != np.ndarray or type(ref) != np.ndarray:
        raise ValueError("Both audio and reference must be nd arrays")

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

    Inputs:
        - values: list type object. Contains every band leq value
    Output:
        - total_value: float type object. Sum of energy bands
    """

    values = np.array(values)
    pressure = 10**(values/10)
    sum_pressure = np.sum(pressure)
    eps = np.finfo(float).eps

    total_value = 10*np.log10(sum_pressure + eps)
    return total_value

def inversesweep(sweep, f1, f2, sr):
    """
    Generate an inverse sweep signal from a given sweep signal.

    Inputs:
        - sweep: ndarray
            The original sweep signal.
        - f1: float
            Start frequency of the sweep signal in Hz.
        - f2: float
            End frequency of the sweep signal in Hz.
        - sr: int
            Sampling rate.

    Output:
        - inverse_sweep_normalized: ndarray
            The generated and normalized inverse sweep signal.

    """
    T = int(len(sweep)/sr)
    t = np.linspace(0, T, T*sr)
    R = np.log(f2/f1)
    envolvente = np.exp((t*R)/T)
    inverse_sweep = sweep[::-1]/envolvente
    inverse_sweep_normalized = inverse_sweep/np.max(np.abs(inverse_sweep))
    
    return inverse_sweep_normalized

def get_rir(audio, sweep_ref, f1,f2, sr=48000):
    """
    Generate a room impulse response (RIR) from recorded audio using a sine sweep reference.
    
    Inputs:
        - audio: ndarray
            Recorded audio signal containing the sine sweep response.
        - sweep_ref: ndarray
            Sine sweep reference signal used for the measurement.
        - f1: float
            Start frequency of the sine sweep in Hz.
        - f2: float
            End frequency of the sine sweep in Hz.
        - sr: int, optional
            Sampling rate. Default is 48000.

    Output:
        - rir_trim: ndarray
            Trimmed and normalized room impulse response (RIR).

    """
    inv_sweep = inversesweep(sweep_ref, f1, f2, sr)

    rir = signal.fftconvolve(audio, inv_sweep, mode='same')
    
    rir_norm = rir / np.max(np.abs(rir))

    max_rir = np.argwhere(rir_norm == rir_norm.max())
    max_rir = max_rir[0]
    max_rir = max_rir[0]
    
    rir_trim = rir_norm[max_rir-20:max_rir+47980]
    
    return rir_trim

def rir_filt(rir, f1=100, f2=5000, sr = 48000):
    """
    Filter a room impulse response (RIR) using a sine sweep filter.

    Inputs:
        - rir: ndarray
            Room impulse response (RIR) to be filtered.
        - f1: float, optional
            Low-pass filter cutoff frequency in Hz. Default is 70 Hz.
        - f2: float, optional
            High-pass filter cutoff frequency in Hz. Default is 6000 Hz.
        - sr: int, optional
            Sampling rate. Default is 48000.

    Output:
        - rir_filtered: ndarray
            Filtered room impulse response (RIR).

    """
    sos = fb.sinesweep_filter(f1, f2, sr)
    rir_filtered = fb.filter_audio(sos, rir)
    
    return rir_filtered

def get_rirs(sinesweeps,  sinesweep_ref, f1, f2, fs):
    """
    From a list of signals it searches for its impulse response (RIR) one by one, and returns a list of the impulse responses

    Inputs:
        - sinesweeps: List of ndarrays
            List of recorded audio signals containing sine sweep responses.
        - sinesweep_ref: ndarray
            Sine sweep reference signal used for the measurements.
        - f1: float
            Start frequency of the sine sweep in Hz.
        - f2: float
            End frequency of the sine sweep in Hz.
        - fs: int
            Sampling rate.

    Output:
        - rirs: list type
            Returns a list of rirs of the input signals.

    """
    rirs = []
    for i in range(len(sinesweeps)):
        #Obtengo el rir de esa posicion
        riri = get_rir(sinesweeps[i], sinesweep_ref, f1, f2, fs)
        #Filtro
        riri = rir_filt(riri)
        rirs.append(riri)
    return rirs

def prom_rirs(rirs):
    """
    Compute the normalized sum of room impulse responses (RIRs).

    Inputs:
        - rirs: list of numpy arrays
            List of room impulse responses (RIRs) as 1D numpy arrays.

    Output:
        - sum_rirs_normalized: numpy array
            The normalized sum of RIRs.

    """
    sum_rirs = np.zeros(rirs[0].size)

    for i in range(len(rirs)):
        sum_rirs += rirs[i]
    sum_rirs_normalized = sum_rirs / np.max(sum_rirs)
    return sum_rirs_normalized

def get_paths(filename, sheet_name):
    """
    Extract file paths from an Excel file.

    Inputs:
        - filename (str): The name of the Excel file.
        - sheet_name (str): The name of the Excel sheet containing paths.

    Output:
        - signals_paths (list of str): A list of file paths extracted from the specified Excel sheet.
    """
    signals_paths = rf.excel_sheets_data_to_DataFrame(filename, sheet_name)
    signals_paths = signals_paths[0]["Path"]
    
    return signals_paths

def get_signals(signals_paths):
    """
    Load audio signals from file paths.

    Inputs:
        - signals_paths (list of str): List of file paths to audio signals.

    Output:
        - signals_sr (list of int): List of sample rates for the loaded audio signals.
        - signals (list of ndarray): List of loaded audio signals.

    """
    signals = []
    signals_sr = []
    
    for i in range(len(signals_paths)):
        signal_i, sr_i = load_audio(signals_paths[i])
        signals.append(signal_i)
        signals_sr.append(sr_i)
    
    return signals_sr, signals    

def aural(audio, rir, fs=48000):
    """
    Auralize audio using a room impulse response (RIR). The input audio should be in mono.

    Inputs:
        - audio: ndarray
            Mono audio signal to auralize.
        - rir: ndarray
            Room impulse response (RIR) used for auralization.
        - fs: int, optional
            Sampling rate. Default is 48000.

    Output:
        - aur: ndarray
            Auralized audio signal.

    """
    #Convoluciono para auralizar
    aur = signal.fftconvolve(audio, rir)
    
    #Normalizo
    aur = aur / np.max(np.abs(aur))  
    
    return aur

def get_sonometer_leq(data, central_freqs, *measurements):
    """
    Get sound level equivalent (Leq) values from a dataset for specified measurements and central frequencies.

    Inputs:
        - data (DataFrame): DataFrame containing sound level data with columns 'Frequency [Hz]' and measurement values.
        - central_freqs (list): List of central frequencies of interest.
        - measurements (variable arguments): Names of the measurements to extract Leq values for.

    Output:
        - leq_values (list): List of Leq values for the specified measurements and central frequencies. If a single measurement is provided, a single list is returned. If multiple measurements are provided, a list of lists is returned.

    """

    if len(measurements) == 1:
        i = measurements[0]
        leq_values = []
        for freq in data["Frequency [Hz]"].values:
            freq = int(np.rint(freq))
            if freq in central_freqs:
                filtro_frecuencias = data["Frequency [Hz]"] == freq
                leq = data.loc[filtro_frecuencias, i].values
                leq_values.append(float(leq))

        return leq_values

    elif len(measurements) > 1:
        all_leqs = []
        for i in measurements:
            leq_values = []
            for freq in data["Frequency [Hz]"].values:
                freq = int(np.rint(freq))
                if freq in central_freqs:
                    filtro_frecuencias = data["Frequency [Hz]"] == freq
                    leq = data.loc[filtro_frecuencias, i]
                    leq_values.append(leq)
            all_leqs.append(leq_values)    

        return all_leqs    