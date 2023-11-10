import audio_functions as auf
import filters_bank as fb
import plot
import codigos_medicion as cm
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

def check_filter_plot(f0, pol_coef, fs, bw, title=False, figsize=False, show=True):
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
    

    b, a = pol_coef

    wn, H = signal.freqz(b,a, worN=512*6)
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

pol_coef_1 = fb.create_octave_filter(1000, 48000, 3)
check_filter_plot(1000, pol_coef_1, 48000,  "octave")

pol_coef_2 = fb.create_third_octave_filter(1000, 48000, 4)
check_filter_plot(1000, pol_coef_2, 48000,  "third")

resp_en_freq_octavas = lambda: plot.plot_ftf([pol_coef_1], 48000, f_lim=[80, 6000], show=False)
atenuacion_octavas = lambda: plot.check_filter_plot(1000, pol_coef_1, 48000,  "octave")
resp_en_freq_tercios = lambda: plot.plot_ftf([pol_coef_2], 48000, f_lim=[60, 6000], show=False)
atenuacion_tercios = lambda: plot.check_filter_plot(1000, pol_coef_2, 48000,  "third")

plot.multiplot(resp_en_freq_octavas, atenuacion_octavas, resp_en_freq_tercios, atenuacion_tercios, figsize=(16, 10))