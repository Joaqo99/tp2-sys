import numpy as np
import soundfile as sf
import plot as plot
import audio_functions as auf


# MIC 3 - FUENTE 2 - SIN GENTE

sweep_path_SG = auf.get_paths("Paths.xlsx", "SS 30 - 8K - F2 - SG")
signal_3, fs_3 = auf.load_audio(sweep_path_SG[2])

sweep_ref, fs = auf.sf.read("audios/Audios de Grabaciones/seno_logaritmico_25seg_30-8K.wav")

rir_3_SG = auf.get_rir(signal_3, sweep_ref, 30, 8000, fs)
rir_3_SG = auf.rir_filt(rir_3_SG)

#Cargo el audio a auralizar
voice, fs= auf.sf.read("audios/anechoic_voice_48.wav")
voice_mono = auf.to_mono(voice)

#Consigo el time array de cada rir para graficar, y grafico
dur_SG, t_rir3_SG = auf.get_audio_time_array(rir_3_SG, fs)

#rir_3_SG es la respuesta al impulso de un recinto. La Paso a dB

rir_3_SG_dB = 10 * auf.np.log10((rir_3_SG)**2)

plot.plot_signal([t_rir3_SG, rir_3_SG_dB], title="Representacion de energía - RIR Mic 3 - Sin Gente", grid=True, figsize=(8,5))