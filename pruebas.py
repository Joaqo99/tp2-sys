import audio_functions as auf
import filters_bank as fb
import plot
import codigos_medicion as cm
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

def check_filter_plot(f0, sos, fs, bw, title=False, figsize=False, show=True):
    """
    Plots the magnitude (in dB) of a filter in frequency respect the attenuation limits.
    Inputs:
        - f0: int type object. central frequency of filter
        - pol_coef: list type object. Filter coefficients
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

    wn, H = signal.sosfreqz(sos, worN=512*6)
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


octave_rel = 2
ref_freq = 1000


def sos_create_octave_filter(f0, fs, order):
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
    sos = signal.butter(order, [fc1, fc2], btype='bandpass', output='sos')
    return sos

def sos_create_third_octave_filter(f0, fs, order):
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
    sos = signal.butter(order, [fc1, fc2], btype='bandpass', output='sos')
    return sos

# VALORES PARA PROBAR
sos_1 = sos_create_octave_filter(1000, 48000, 3)
check_filter_plot(1000, sos_1, 48000,  "octave")

sos_2 = sos_create_third_octave_filter(250, 48000, 4)
check_filter_plot(250, sos_2, 48000,  "third")
