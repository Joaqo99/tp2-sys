import audio_functions as auf
import filters_bank as fb
import plot
import codigos_medicion as cm
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

octave_bands_leq_m1_nf = []
octave_bands_leq_m6_nf = []
octave_bands_leq_s1_nf = []
octave_bands_leq_s2_nf = []

for i in range(0, 18, 3):
    octave_s1 = auf.sum_bands([thirds_bands_leq_s1_nf[i], thirds_bands_leq_s1_nf[i + 1], thirds_bands_leq_s1_nf[i + 2]])
    octave_s2 = auf.sum_bands([thirds_bands_leq_s2_nf[i], thirds_bands_leq_s2_nf[i + 1], thirds_bands_leq_s2_nf[i + 2]])
    octave_m1 = auf.sum_bands([thirds_bands_leq_m1_nf[i], thirds_bands_leq_m1_nf[i + 1], thirds_bands_leq_m1_nf[i + 2]])
    octave_m6 = auf.sum_bands([thirds_bands_leq_m6_nf[i], thirds_bands_leq_m6_nf[i + 1], thirds_bands_leq_m6_nf[i + 2]])

    octave_bands_leq_s1_nf.append(octave_s1)
    octave_bands_leq_s2_nf.append(octave_s2)
    octave_bands_leq_m1_nf.append(octave_m1)
    octave_bands_leq_m6_nf.append(octave_m6)