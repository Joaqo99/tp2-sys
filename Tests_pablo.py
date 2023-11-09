import numpy as np
import soundfile as sf
import plot as plot
import audio_functions as auf


# MIC 3 - FUENTE 2 - SIN GENTE

sweep_path_SG = auf.get_paths("Paths.xlsx", "SS 30 - 8K - F2 - SG")
signal_3, fs_3 = auf.load_audio(sweep_path_SG[2])

sweep_ref, fs = sf.read("./audios/Audios de Grabaciones/seno_logaritmico_25seg_30-8K.wav")

rir_3_SG = auf.get_rir(signal_3, sweep_ref, 30, 8000, fs)
rir_3_SG = auf.rir_filt(rir_3_SG)

#Cargo el audio a auralizar
voice, fs= sf.read("./audios/anechoic_voice_48.wav")
voice_mono = auf.to_mono(voice)

#Auralizacion del mic 3 SG
aur_voice_3_SG = auf.aural(voice_mono, rir_3_SG, fs)

#Consigo el time array de cada rir para graficar, y grafico
dur_SG, t_rir_SG = auf.get_audio_time_array(rir_3_SG, fs)


plot.plot_signal([t_rir_SG, rir_3_SG], title="Representacion temporal de RIR Sin Gente", grid = True)

#Plot OPCIONAL de respuesta en frecuencia de otro módulo (funcion pablo)
#plot.plot_fft(rir_3_SG, fs, N=50)

#------------------------------------------------------------------------------------------------------------------------

# MIC 3 - FUENTE 2 - CON GENTE

sweep_path_CG = auf.get_paths("Paths.xlsx", "SS 30 - 8K - F2 - CG")
signal_3, fs_3 = auf.load_audio(sweep_path_CG[2])


sweep_ref, fs = sf.read("./audios/Audios de Grabaciones/seno_logaritmico_25seg_30-8K.wav")

rir_3_CG = auf.get_rir(signal_3, sweep_ref, 30, 8000, fs)
rir_3_CG = auf.rir_filt(rir_3_CG)

#Auralizacion del mic 3 CG
aur_voice_3_CG = auf.aural(voice_mono, rir_3_CG, fs)

#Consigo el time array de cada rir para graficar
dur_CG, t_rir_CG = auf.get_audio_time_array(rir_3_CG, fs)

plot.plot_signal([t_rir_CG, rir_3_CG], title="Representacion temporal de RIR Con Gente", grid = True)

#Plot OPCIONAL de respuesta en frecuencia de otro módulo (funcion pablo)
#plot.plot_fft(rir_3_CG,fs,N=50)

#------------------------------------------------------------------------------------------------------------------------

#Plots para diferenciar SG y CG - Los "t" son iguales para el plot temporal
#Sumo los rir en listas
rir_CG_SG=[]
rir_CG_SG = [rir_3_SG, rir_3_CG]

plot.rir_subplot(rir_CG_SG, t_rir_CG, plot_type="SG-CG") 

#Grafico la respuesta en frecuencia superpuesta

plot.plot_mult_fft(rir_3_SG, rir_3_CG, fs, N = 50, title="Comparacion de Respuesta en Frecuencia entre RIR SG y CG")
