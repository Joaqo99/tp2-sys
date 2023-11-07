import pandas as pd
import numpy as np

def gen_sin_table(duracion, frecuencia, tabla_seno):
    """
    Generate a sinusoidal signal using a pre-defined lookup table.
    
    Parameters:
    -----------
    duracion : int or float
        Duration of the signal in seconds.
    frecuencia : int or float
        Frequency of the sinusoidal signal in Hz.
    tabla_seno : pandas.DataFrame
        DataFrame containing a lookup table with "idx" and "val" columns.
    
    Returns:
    --------
    tiempo : numpy.ndarray
        Array containing the time values.
    amplitudes : numpy.ndarray
        Array containing the amplitudes of the generated sinusoidal signal.
        
    This function generates a sinusoidal signal by looking up amplitudes from the provided lookup table for the specified frequency and duration.
    
    """
    index = tabla_seno["idx"]
    value = tabla_seno["val"]
    
    sr=len(index)
    
    tiempo = np.linspace(0,duracion, duracion*sr)
    
    n = np.arange(0, duracion*sr)
    amp_index = np.mod(n*frecuencia, sr)
    amplitudes = value[amp_index]

    return tiempo, amplitudes