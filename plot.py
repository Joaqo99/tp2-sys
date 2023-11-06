from matplotlib import pyplot as plt
from scipy import signal
import numpy as np

def plot_signal(*vectors, xticks=None, yticks=None, title=None, file_name=False, grid=False, log=False, figsize=False, x_type="Sample", dB=False, show=True):
    """
    Plots a signal.
    Input:
        - n: array type object. Sample/time vector.
        - signal: array type object. Signal vector.    
        - xticks: Optional. 
            - If signal is Sample type, array type object. 
            - If signal is Time type, int type object. 
        - yticks: array type object. Optional
        - title: string type object. Optional
        - file_name: string type object. Optional. If true, saves the figure in graficos folder.
        - grid: boolean type object. Optional.
        - log: boolean type object. Optional.
        - figsize: tuple of ints type object. Optional.
        - x_type: string type object. Optional. If the x values are time or samples. "Sample" by default. 
            - Allowed values: Sample, Time.
        - dB: Bool type object. Optional, false by default. If true, the amplitude is in dB scale.
    Output:
        - Signal plot
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize) 

    #Plot type and xticks depending on signal type
    if x_type == "Sample":
        for vector in vectors:
            n, signal = vector
            plt.stem(n, signal)
            plt.xlabel("Sample")
            if type(xticks) == np.ndarray:
                plt.xticks(xticks)
            elif xticks != None:
                raise Exception("If signal is Sample type, xticks value must be a numpy array")
        
    elif x_type == "Time":
        for vector in vectors:
            n, signal = vector
            plt.plot(n, signal)
            plt.xlabel("Time [s]")
            if type(xticks) == int:
                plt.xticks(np.arange(0, xticks, 1))
            elif type(xticks) != None:
                raise Exception("If signal is Time type, xtick value must be an int")
        
    else:
        raise Exception("No valid type")
    
    if dB:
        plt.ylabel("Amplitude [dB]")
    else:
        plt.ylabel("Amplitude")

    if type(yticks) == np.ndarray:
        plt.yticks(yticks)
    plt.grid(grid)

    if log:
        plt.yscale("log")
        plt.ylabel("Amplitude (logarithmic)")
        if dB:
            plt.ylabel("Amplitude (logarithmic) [dB]")

    if title:
        plt.title(title)

    #save file
    if file_name:
        plt.savefig(f"../graficos/{file_name}.png")
        #print(f"File saved in graficos/{file_name}.png")
    
    if show: 
        plt.show()
    else:
        plt.ioff()

def plot_spectre(*audio_signals, fs, N = 1, f_lim = False, title = "Espectro en frecuencia", show=True, figsize=False):
    """
    Generates and displays a graph of the frequency spectrum of an audio signal.
    
    Parameters
    ----------
    - audio_signal : ndarray
        Array containing the audio signal to be plot.
    - sample_rate : int
        Sampling rate.
    - N : int,float
        Parameter of the window size for moving average filter
    - titulo: str
        Optional title for the plot. Default value: "Espectro en frecuencia".
        
    Returns
    -------
    - None
    
    Raises
    ------
    ValueError
        Checks if the N value is greater than cero.

    """
    #Verifico si la señal ingresada es estéreo o mono
    for audio_signal in audio_signals:
        if len(audio_signal.shape) > 1:
            #La transformo en una señal mono
            audio_signal = (audio_signal[:,0]/2)+(audio_signal[:,1]/2)

        # Calcula el espectro de amplitud de la señal
        spectrum = np.fft.fft(audio_signal)
        fft = spectrum[:len(spectrum)//2]

        # Encuentra el rango de frecuencias positivas
        fft_mag = abs(fft) / len(fft)
        freqs = np.linspace(0, fs/2, len(fft))

        # Paso la magnitud a decibeles
        fft_mag_norm = fft_mag / np.max(abs(fft_mag))
        eps = np.finfo(float).eps
        fft_mag_db = 20 * np.log10(fft_mag_norm + eps)

        # Aplico el filtro media movil
        ir = np.ones(N)*1/N # respuesta al impulso de MA
        smoothed_signal = signal.fftconvolve(fft_mag_db, ir, mode='same')

        # Escala logarítmica para el eje x
        plt.semilogx(freqs, smoothed_signal, color='g')

    if figsize:
        plt.figure(figsize=figsize) 

    if f_lim:
        f_min, f_max = f_lim
        plt.xlim(f_min, f_max)
        xticks =list(filter(lambda f: f >= f_min and f <= f_max, oct_central_freqs))
        plt.xticks(xticks, xticks)
    
    plt.ylim(-60, np.max(fft_mag_db) + 10)
    plt.xlabel("Frecuencia [Hz]", fontsize=12)
    plt.ylabel("Amplitud [dB]", fontsize=12)
    plt.title(title, fontsize=16)
    plt.grid(True)

    if show: 
        plt.show()
    else:
        plt.ioff()

def plot_ftf(filters, fs, f_lim=False, figsize=False, show=True):
    """
    Plots the filter transfer function
    Input:
        - filters: list of filters.
        - fs: int type object. sample rate
        - f_lim: list type object. Frequency visualization limits. False 
    """
    if figsize:
        plt.figure(figsize=(10, 6))
    for pol_coef in filters:
        b, a = pol_coef
        wn, H = signal.freqz(b,a, worN=8192)
        f= (wn*(0.5*fs))/np.pi

        eps = np.finfo(float).eps

        H_mag = 20 * np.log10(abs(H) + eps)


        # #La magnitud de H se grafica usualmente en forma logarítmica
        plt.semilogx(f, H_mag)
    plt.ylabel('Mag( dB )')
    plt.xlabel('Frecuencia (Hz)')

    if f_lim:
        f_min, f_max = f_lim
        plt.xlim(f_min, f_max)
        xticks =list(filter(lambda f: f >= f_min and f <= f_max, oct_central_freqs))
        plt.xticks(xticks, xticks)

    plt.ylim(-6,1)
    plt.title('Magnitud de H')
    plt.ylabel('Mag( dB )')
    plt.legend()
    plt.grid()
    if show: 
        plt.show()
    else:
        plt.ioff()

def save_signal(n, signal, filename):
    """
    Wraps a time/sample and signal vectors into a package to save it as .npy
    Input:
        - n: array type object. Time/sample array.
        - signal: array type object. Signal array.
        - filename: str.
    """
    pack = np.stack((n, signal))
    np.save(f'audios/{filename}.npy', pack)
    print(f"Señal guardado en ./audios/{filename}.npy")

def load_signal(filename):
    """
    Wraps a time/sample and signal vectors into a package to save it as .npy
    Input:
        - filename: str.
    Ouput:
        - n: array type object. Time/sample array.
        - signal: array type object. Signal array.
    """
    pack = np.load(f'audios/{filename}.npy')
    n, signal = pack
    return n, signal

def plot_leqs(*signals, x=[], title=False, figsize=False, show=True, rotate=False, info_type="Frequency"):
    """
    Plot a variable number of plots of leq values in rows of two plots by row.
    
    Input:
        - signals : Optional amount of values. For each signal: Dict type object. Must contain:
            - leq: list of leq values.
            - label: string type object.
        - freqs: list of central frequency. Central frequencies of multiple signals over the same axis must be the same.
        - titles: Optional dictionary for subplot titles. Keys are subplot numbers (ax) and values are titles.
    """
    if figsize:
        plt.figure(figsize=figsize)

    if info_type=="Frequency":
        x = [str(np.rint(valor)) for valor in x]
    elif info_type == "categories":
        pass
    else:
        raise ValueError("Not valid info_type value") 
    
    #import pdb; pdb.set_trace()

    for signal in signals:
        if "label" in signal.keys():
            plt.bar(x, signal["leq"], label=signal['label'], alpha=0.7)
        else:
            plt.bar(x, signal["leq"], alpha=0.7)

        plt.xlabel("Frecuencias centrales [Hz]")
        if rotate:
            plt.xticks(rotation=45)
        plt.ylabel("Nivel de energía equivalente [dB]")
        plt.legend()
        plt.grid()

    plt.tight_layout()

    if title:
        plt.title(title)

    if show: 
        plt.show()
    else:
        plt.ioff()

def multiplot(*plots, figsize=(8, 5)):
    """
    Receive single plots as lambda functions and subplots them all.
    """
    num_plots = len(plots)
    rows = (num_plots + 1)//2
    plt.figure(figsize=figsize)
    for i, figure in enumerate(plots):
        plt.subplot(rows,2, i + 1)
        figure()

    plt.show()




oct_central_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
thirds_central_freqs = [78.745, 99.123, 125.0, 157.49, 198.43, 250.0, 314.98, 396.85, 500.0, 628.96, 793.7, 1000.0, 1259.9, 1587.4, 2000.0, 2519.8, 3174.8, 4000.0, 5039.7]