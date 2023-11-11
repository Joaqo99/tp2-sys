from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy import signal

import numpy as np

def plot_signal(*vectors, xticks=None, yticks=None, title=None, file_name=False, grid=False, log=False, figsize=False, dB=False, show=True, plot_type=None):
    """
    Plots a signal.
    Input:
        - n: array type object. Sample/time vector.
        - signal: array type object. Signal vector.    
        - xticks: Optional. 
        - yticks: array type object. Optional
        - title: string type object. Optional
        - file_name: string type object. Optional. If true, saves the figure in graficos folder.
        - grid: boolean type object. Optional.
        - log: boolean type object. Optional.
        - figsize: tuple of ints type object. Optional.
        - dB: Bool type object. Optional, false by default. If true, the amplitude is in dB scale.
        - plot_type: str type. Type of signal to show. Can be a ED or temporal graph.
    Output:
        - Signal plot
        - If file_name is true, saves the figure and prints a message.
    """
    if figsize:
        plt.figure(figsize=figsize) 

    if type(xticks) != int and type(xticks) != type(None):            
            raise Exception("xtick value must be an int")
    
    
    if type(xticks) == int:
        if xticks == 1:
            plt.xticks(np.arange(0, xticks, 0.1))
        else:
            plt.xticks(np.arange(0, xticks+1, 1))

    for vector in vectors:
        n, signal = vector
        plt.plot(n, signal)
        plt.xlabel("Tiempo [s]")
        # Agregar líneas en el eje y y x a -60dB
        if plot_type == "ED":
            plt.axhline(y=-60, color='r', linestyle='--', label='-60 dB Threshold')  # Línea horizontal
            plt.axvline(x=0.47, color='g', linestyle='--', label='Crossing Time')  # Línea vertical
    
    if dB:
        plt.ylabel("Amplitud [dB]")
    else:
        plt.ylabel("Amplitud")

    if type(yticks) == np.ndarray:
        if type(yticks) != np.ndarray:            
            raise Exception("ytick value must be an array")
        plt.ylim(np.min(yticks), np.max(yticks))
        plt.yticks(yticks)
    plt.grid(grid)
    
    if log:
        plt.yscale("log")
        plt.ylabel("Amplitud (logarithmic)")
    if dB:
        plt.ylabel("Amplitud [dB]")

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

def plot_ftf(filters, fs, f_lim=False, figsize=False, show=True, title=False):
    """
    Plots the filter transfer function
    Input:
        - filters: list of filters. They must be in sos format
        - fs: int type object. sample rate
        - f_lim: list type object. Frequency visualization limits. False 
    """
    if figsize:
        plt.figure(figsize=figsize)
    for sos in filters:
        wn, H = signal.sosfreqz(sos, worN=4096)
        f= (wn*(0.5*fs))/np.pi

        eps = np.finfo(float).eps

        H_mag = 20 * np.log10(abs(H) + eps)


        # #La magnitud de H se grafica usualmente en forma logarítmica
        plt.semilogx(f, H_mag)
    plt.xlabel('Frecuencia (Hz)')

    if f_lim:
        f_min, f_max = f_lim
        plt.xlim(f_min, f_max)
        xticks =list(filter(lambda f: f >= f_min and f <= f_max, oct_central_freqs))
        plt.xticks(xticks, xticks)

    plt.ylim(-6,1)
    if title:
        plt.title(title)

    plt.ylabel('Magnitud [dB]')
    plt.grid()
    if show: 
        plt.show()
    else:
        plt.ioff()

def check_filter_plot(f0, sos, fs, bw, title=False, figsize=False, show=True):
    """
    Plots the magnitude (in dB) of a filter in frequency respect the attenuation limits.
    Inputs:
        - f0: int type object. central frequency of filter
        - sos: array type object. Second order sections of the filter.
        - fs: int type object. sample rate
        - bw: str type object. Bandwidth of filter. Two possible values:
            - octave
            - third
        - title: string type object. Optional, false by default.
    """
    if figsize:
        plt.figure(figsize=figsize)
    G = 2
    f_lims = np.array([G**(-3), G**(-2), G**(-1), G**(-1/2), G**(-3/8), G**(-1/4), G**(-1/8), 1, G**(1/8), G**(1/4), G**(3/8), G**(1/2), G, G**2, G**3])
    lim_inf = [-200.0, -180.0, -80.0, -5.0, -1.3, -0.6, -0.4, -0.3, -0.4, -0.6, -1.3, -5.0, -80.0, -180.0, -200.0]
    lim_sup = [-61.0, -42.0, -17.5, -2.0, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, -2.0, -17.5, -42.0, -61.0]

    if bw == "octave":
        x_ticks = [G**(-3), G**(-2), G**(-1), G**(-1/2), 1, G**(1/2), G, G**2, G**3]
        xtick_labels = [
            r'$G^{-3}$',
            r'$G^{-2}$',
            r'$G^{-1}$',
            r'$G^{-\frac{1}{2}}$',
            r'$1$',
            r'$G^{\frac{1}{2}}$',
            r'$G$',
            r'$G^2$',
            r'$G^3$'
        ]

        xlim_a, xlim_b = 0.1, 10
        minor_ticks = False

    elif bw == "third":
        f_lims = f_lims**(1/3)

        x_ticks = np.array([G**(-3), G**(-2), G**(-1), G**(-1/2), 1, G**(1/2), G, G**2, G**3])
        x_ticks = list(x_ticks**(1/3))

        xtick_labels = [
            r'$G^{-1}$',
            r'$G^{-\frac{2}{3}}$',
            r'$G^{-\frac{1}{3}}$',
            r'$G^{-\frac{1}{6}}$',
            r'$1$',
            r'$G^{\frac{1}{6}}$',
            r'$G^{\frac{1}{3}}$',
            r'$G^{\frac{2}{3}}$',
            r'$G^{1}$'
        ]

        xlim_a, xlim_b = 0.5, 2
        minor_ticks = True
    else:
        raise ValueError('No valid bw input. Values must be "octave" or "third"')

    wn, H = signal.sosfreqz(sos, worN=4096)
    f= (wn*(0.5*fs))/(np.pi*f0)

    eps = np.finfo(float).eps

    H_mag = 20 * np.log10(abs(H) + eps)

    plt.semilogx(f, H_mag, label="Filtro", color="#030764") 
    plt.semilogx(list(f_lims), lim_sup, label="Lim. sup. de atenuación", linestyle='dashed', color="#c20078") 
    plt.semilogx(list(f_lims), lim_inf, label="Lim. inf. de atenuación", linestyle='dashed', color="red")
    plt.xticks(x_ticks, xtick_labels, minor=minor_ticks)
    plt.xlim(xlim_a, xlim_b)
    plt.ylim(-60, 1)
    plt.legend()
    
    if bw== "third":
        plt.grid(which='both')
    else:
        plt.grid()

    if title:
        plt.title(title)

    plt.xlabel("Frecuencia normalizada")
    plt.ylabel("Amplitud [dB]")

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

def plot_leqs(x, *signals, title=False, figsize=False, show=True, rotate=False, info_type="frequency"):
    """
    Plot a variable number of plots of leq values in rows of two plots by row.
    
    Input:
        - x: list type object. List x-axis values
        - signals : Optional amount of values. For each signal: Dict type object. Must contain:
            - leq: list of leq values.
            - label: string type object.
            - color: string type object.
        - freqs: list of central frequency. Central frequencies of multiple signals over the same axis must be the same.
        - titles: Optional dictionary for subplot titles. Keys are subplot numbers (ax) and values are titles.
        - show: Bool type object. True by default. Shows the plot
        - rotate: Bool type object. True by default. Rotates 45º the x-axis values
        - info_type: 2 posible values: "frequency" or "categories". Frequency by de
            - 
    """
    if type(x) != list:
        raise ValueError("x must be a list")
    if figsize:
        plt.figure(figsize=figsize)

    if info_type=="frequency":
        x = [str(np.rint(valor)) for valor in x]
    elif info_type == "categories":
        pass
    else:
        raise ValueError("Not valid info_type value") 
    
    #import pdb; pdb.set_trace()

    for signal in signals:
        label = signal["label"] if "label" in signal.keys() else None
        color = signal["color"] if "color" in signal.keys() else None

        plt.bar(x, signal["leq"], label=label, color=color, alpha=0.7)


        if info_type=="Frequency":
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
    Inputs:
        plots: lambda function type object.
    """
    num_plots = len(plots)
    rows = (num_plots + 1)//2
    plt.figure(figsize=figsize)
    for i, figure in enumerate(plots):
        plt.subplot(rows,2, i + 1)
        figure()

    plt.show()

def plot_fft(audio_signal, sample_rate=48000, N=10, title="Frequency Spectrum"):
    """
    Generates and displays a graph of the frequency spectrum of an audio signal.

    Parameters:
        - audio_signal: ndarray
            Array containing the audio signal to be plotted.
        - sample_rate: int
            Sampling rate. Default value: 48000.
        - N: int or float
            Window size parameter for the moving average filter. Default value: 10.
        - title: str
            Optional title for the plot. Default value: "Frequency Spectrum".

    Returns:
        - None

    Raises:
        - ValueError
            Checks if the 'N' value is greater than zero.

    """
    # Verify if the input signal is stereo or mono
    if len(audio_signal.shape) > 1:
        # Convert it to a mono signal
        audio_signal = (audio_signal[:, 0] / 2) + (audio_signal[:, 1] / 2)

    # Calculate the amplitude spectrum of the signal
    spectrum = np.fft.fft(audio_signal)
    fft = spectrum[:len(spectrum) // 2]

    # Find the range of positive frequencies
    fft_mag = abs(fft) / len(fft)
    freqs = np.linspace(0, sample_rate / 2, len(fft))

    # Convert magnitude to decibels
    fft_mag_norm = fft_mag / np.max(abs(fft_mag))
    eps = np.finfo(float).eps
    fft_mag_db = 20 * np.log10(fft_mag_norm + eps)

    # Apply the moving average filter
    ir = np.ones(N) * 1 / N  # Moving average impulse response
    smoothed_signal = signal.fftconvolve(fft_mag_db, ir, mode='same')

    # Logarithmic scale for the x-axis
    plt.figure(figsize=(12, 5))
    plt.semilogx(freqs, smoothed_signal, color='g')
    ticks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
    plt.xlim(20, 22000)
    plt.ylim(-80, np.max(fft_mag_db) + 10)
    plt.xlabel("Frecuencia [Hz]", fontsize=14)
    plt.ylabel("Amplitud [dB]", fontsize=14)
    if N != 1:
        plt.title(f"{title} - Filter Window = {N}", fontsize=16)
    else:
        plt.title(title, fontsize=16)
    plt.grid(True)

    # Display the plot
    plt.show()

def plot_rir_casos(rir_casos, fs=48000):
    """
    Plot the overlaid frequency responses of different room impulse response (RIR) cases.

    Parameters:
        - rir_casos (list of ndarrays): List of RIRs to plot.
        - fs (int, optional): Sampling rate. Default is 48000.

    Returns:
        None

    """
    for i, func in enumerate(rir_casos):
    
        if i == 0:
            plt.figure(figsize=(12, 5))
        # Calcula el espectro de amplitud de la señal
        spectrum = np.fft.fft(func)
        fft = spectrum[:len(spectrum)//2]
    
        # Encuentra el rango de frecuencias positivas
        fft_mag = abs(fft) / len(fft)
        freqs = np.linspace(0, fs/2, len(fft))
    
        # Paso la magnitud a decibeles
        fft_mag_norm = fft_mag / np.max(abs(fft_mag))
        eps = np.finfo(float).eps
        fft_mag_db = 10 * np.log10((fft_mag_norm + eps)**2)
            
        # Escala logarítmica para el eje x
        plt.semilogx(freqs, fft_mag_db, "g")
        
        
    ticks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
    plt.xlim(20, 22000)
    plt.ylim(-80, np.max(fft_mag_db) + 10)
    plt.xlabel("Frecuencia [Hz]", fontsize=14)
    plt.ylabel("Amplitud [dB]", fontsize=14)
    # if N != 1:
    #     plt.title(f"{title} - Ventana del filtro = {N}", fontsize=16)
    # else:
    #     plt.title(title, fontsize=16)
    plt.title(f"Respuesta en frecuencia superpuesta", fontsize=16)
    plt.grid(True)
    
    
    # Muestra el gráfico
    plt.show()

    return

def rir_subplot(rir_list, t, plot_type="12-RIR", case=None, title=None):
    """
    Plot room impulse responses (RIRs) in a subplot arrangement.

    Parameters:
        - rir_list (list of ndarrays): List of RIRs to plot.
        - t (ndarray): Time values corresponding to the RIRs.
        - plot_type (str, optional): Type of plot. 
        Options are "SG-CG" for comparison between MIC 3 SG and CG, "2-RIR" for for comparison between 2 RIRs 
        or"12-RIR" for comparison between 12 RIRs. Default is "12-RIR".
        - case: Choose between SG or CG in "12-RIR" plot type. Default is not used.
        - title: list type. Optional. Add a title to each plot separately

    Returns:
        None
    """
    if plot_type == "12-RIR":
        fig = plt.figure(figsize=(15, 8))
    else:
        fig = plt.figure(figsize=(10, 3))
    
    # Iterar a través de las funciones y agregar cada una al subplot
    if len(rir_list) == 2:
        for i, func in enumerate(rir_list):
            # Crear un subplot en la posición (n_filas, n_columnas, índice)
            plt.subplot(1, 2, i + 1)
            plt.plot(t, func, "g")
            if plot_type == "SG-CG":
                if i==0:
                    plt.title(f"RIR | MIC 3 | SG")
                else:
                    plt.title(f"RIR | MIC 3 | CG")
            else:
                if title:
                    plt.title(title[i])
            plt.ylabel("Amplitud", fontsize=13)
            plt.xlabel("Tiempo[s]", fontsize=13)
            plt.grid()
    else:                       # 12 - RIRs
        for i, func in enumerate(rir_list):
            # Crear un subplot en la posición (n_filas, n_columnas, índice)
            plt.subplot(4, 3, i + 1)
            plt.plot(t, func, "g")
            if i+1 <= 6:
                plt.title(f"RIR posicion {i+1} | F1 | {case}")
            else:
                plt.title(f"RIR posicion {(i+1)- 6} | F2 | {case}")
            plt.ylabel("Amplitud", fontsize=13)
            plt.xlabel("Tiempo[s]", fontsize=13)
            plt.grid()
    plt.tight_layout()
    plt.show()
    return

def plot_mult_fft(audio_signal1, audio_signal2, fs=48000, N=10, title="Espectro en frecuencia"):
    """
    Generates and displays a graph of the frequency spectrum of audio signals.
    Parameters
    ----------
    - audio_signal1, audio_signal2 : ndarray
        Arrays containing the audio signals to be plotted.
    - fs : int
        Sampling rate.
    - N : int, float
        Parameter of the window size for the moving average filter.
    - title : str
        Optional title for the plot. Default value: "Espectro en frecuencia".

    Returns
    -------
    - None

    Raises
    ------
    - ValueError
        Checks if the N value is greater than zero.

    """
    # Verifico si las señales ingresadas son estéreo o mono
    if len(audio_signal1.shape) > 1:
        # Las transformo en señales mono
        audio_signal1 = (audio_signal1[:, 0] / 2) + (audio_signal1[:, 1] / 2)
    if len(audio_signal2.shape) > 1:
        audio_signal2 = (audio_signal2[:, 0] / 2) + (audio_signal2[:, 1] / 2)

    # Calcula el espectro de amplitud de las señales
    spectrum1 = np.fft.fft(audio_signal1)
    fft1 = spectrum1[:len(spectrum1) // 2]
    spectrum2 = np.fft.fft(audio_signal2)
    fft2 = spectrum2[:len(spectrum2) // 2]

    # Encuentra el rango de frecuencias positivas
    fft_mag1 = abs(fft1) / len(fft1)
    freqs1 = np.linspace(0, fs / 2, len(fft1))
    fft_mag2 = abs(fft2) / len(fft2)
    freqs2 = np.linspace(0, fs / 2, len(fft2))

    # Paso la magnitud a decibeles
    fft_mag_norm1 = fft_mag1 / np.max(abs(fft_mag1))
    eps = np.finfo(float).eps
    fft_mag_db1 = 20 * np.log10(fft_mag_norm1 + eps)
    fft_mag_norm2 = fft_mag2 / np.max(abs(fft_mag2))
    fft_mag_db2 = 20 * np.log10(fft_mag_norm2 + eps)

    # Aplico el filtro media movil
    ir = np.ones(N) * 1 / N  # respuesta al impulso de MA
    smoothed_signal1 = signal.fftconvolve(fft_mag_db1, ir, mode='same')
    smoothed_signal2 = signal.fftconvolve(fft_mag_db2, ir, mode='same')

    # Escala logarítmica para el eje x
    plt.figure(figsize=(12, 5))
    plt.semilogx(freqs1, smoothed_signal1, color='g', label='RIR SG')
    plt.semilogx(freqs2, smoothed_signal2, color='orange', label='RIR CG')
    ticks = [31, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    plt.xticks([t for t in ticks], [f'{t}' for t in ticks])
    plt.xlim(20, 22000)
    plt.ylim(-80, max(np.max(fft_mag_db1), np.max(fft_mag_db2)) + 10)
    plt.xlabel("Frecuencia [Hz]", fontsize=14)
    plt.ylabel("Amplitud [dB]", fontsize=14)
    if N != 1:
        plt.title(f"{title} - Ventana del filtro = {N}", fontsize=16)
    else:
        plt.title(title, fontsize=16)
    plt.grid(True)
    plt.legend()

    # Muestra el gráfico
    plt.show()
    return

def cont_signal_ploter(t, signal, labels = ("Tiempo [s]","Amplitud"), xlimits = (None, None), ylimits = (None, None), xscale = "linear", yscale = "linear", x_mathtext = False, y_mathtext = False, x_ticks = (None), y_ticks = (None), title = "", col = "#1f77b4", grid = True, size = (8, 4), save = False, file_name = "mi_grafico", image_dpi = 200):
    """
    Plot a continuous signal.

    Parameters:
    -----------
    t : numpy.ndarray
        Array of time values
    signal : numpy.ndarray
        Array containing the signal values
    labels : tuple, optional
        Labels for the x and y axes (default is ("Tiempo [s]", "Amplitud"))
    xlimits : tuple, optional
        Tuple specifying the minimum and maximum values for the x-axis
    ylimits : tuple, optional
        Tuple specifying the minimum and maximum values for the y-axis
    xscale : str, optional
        Scaling type for the x-axis (default is "linear")
    yscale : str, optional
        Scaling type for the y-axis (default is "linear")
    x_mathtext : bool, optional
        Use MathText for the x-axis labels (default is False)
    y_mathtext : bool, optional
        Use MathText for the y-axis labels (default is False)
    title : str, optional
        Title of the plot (default is "")
    col : str, optional
        Color of the signal plot (default is "#1f77b4").
    grid : bool, optional
        Enable grid lines on the plot (default is True).
    size : tuple, optional
        Size of the plot figure in inches (default is (8, 4)).
    save : bool, optional
        Save the plot as an image file (default is False).
    file_name : str, optional
        Name of the saved image file (default is "my_plot").
    image_dpi : int, optional
        DPI (dots per inch) for the saved image file (default is 200).

    Returns:
    --------
    None
    """
    x, y = labels
    xlim_min, xlim_max = xlimits
    ylim_min, ylim_max = ylimits
    x_fig_size, y_fig_size = size
    
    fig, ax = plt.subplots()
    
    fig.set_size_inches(x_fig_size, y_fig_size)
    
    ax.plot(t, signal, color = col)
        
    if grid == True:
        plt.grid(alpha = 0.7)
    elif grid == False:
        pass
    if x_mathtext == True:
        ax.ticklabel_format(axis = "x", scilimits = (0, 0), useLocale = True, useMathText = True)
    else:
        pass
    if y_mathtext == True:
        ax.ticklabel_format(axis = "y", scilimits = (0,0), useLocale = True, useMathText = True)
    else:
        pass
    if xlim_min != None or xlim_max != None:
        ax.set_xlim(xlim_min, xlim_max)
    else:
        ax.set_xlim(auto = True)
    
    if ylim_min != None or ylim_max != None:
        ax.set_ylim(ylim_min, ylim_max)
    else:
        ax.set_ylim(auto = True)
    
    if y_ticks != None:
        ax.set_yticks(y_ticks)
    else:
        pass
    
    if x_ticks != None:
        ax.set_xticks(x_ticks)
    else:
        pass
    
    ax.xaxis.set_major_locator(MaxNLocator(nbins = "auto", integer=True, steps = [1, 2, 5, 10]))
    
    plt.xscale(xscale)
    plt.yscale(yscale)

    ax.set_title(title, fontsize=16)
    
    ax.set_xlabel(x, fontsize=12)
    ax.set_ylabel(y, fontsize=12)
    
    if save == True:
        plt.savefig(f'{file_name}.png', dpi = image_dpi)
    
    plt.show()
    
    return

oct_central_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
thirds_central_freqs = [78.745, 99.123, 125.0, 157.49, 198.43, 250.0, 314.98, 396.85, 500.0, 628.96, 793.7, 1000.0, 1259.9, 1587.4, 2000.0, 2519.8, 3174.8, 4000.0, 5039.7]