import audio_functions as auf
import filters_bank as fb
import plot
import data_analysis_functions as daf
import numpy as np
import matplotlib.pyplot as plt

nominal_oct_central_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
exact_oct_central_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
nominal_thirds_central_freqs = [100, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0]
exact_thirds_central_freqs = [99.123, 125.0, 157.49, 198.43, 250.0, 314.98, 396.85, 500.0, 628.96, 793.7, 1000.0, 1259.9, 1587.4, 2000.0, 2519.8, 3174.8, 4000.0, 5039.7]

octave_filters_bank = []
thirds_filters_bank = []

for f in exact_oct_central_freqs:
    pol_coef = fb.create_octave_filter(f, 48000)
    octave_filters_bank.append(pol_coef)

for f in exact_thirds_central_freqs:
    pol_coef = fb.create_third_octave_filter(f, 48000)
    thirds_filters_bank.append(pol_coef)

tono_calibrador_1, tono_calibrador_1_fs = auf.load_audio("1 - CALIBRACION DE MICROFONOS A 94dB SPL/MIC 1 - Earthworks M30_01.wav")
tono_calibrador_2, tono_calibrador_2_fs = auf.load_audio("1 - CALIBRACION DE MICROFONOS A 94dB SPL/MIC 2 - Earthworks M30_02.wav")
tono_calibrador_3, tono_calibrador_3_fs = auf.load_audio("1 - CALIBRACION DE MICROFONOS A 94dB SPL/MIC 3 - Behringer ecm 8000_04.wav")
tono_calibrador_4, tono_calibrador_4_fs = auf.load_audio("1 - CALIBRACION DE MICROFONOS A 94dB SPL/MIC 4 - Behringer ecm 8000_03.wav")
tono_calibrador_5, tono_calibrador_5_fs = auf.load_audio("1 - CALIBRACION DE MICROFONOS A 94dB SPL/MIC 5 - Behringer ecm 8000_03.wav")
tono_calibrador_6, tono_calibrador_6_fs = auf.load_audio("1 - CALIBRACION DE MICROFONOS A 94dB SPL/MIC 6 - Behringer ecm 8000_03.wav")

calibraciones = {
    "1":tono_calibrador_1,
    "2":tono_calibrador_2,
    "3":tono_calibrador_3,
    "4":tono_calibrador_4,
    "5":tono_calibrador_5,
    "6":tono_calibrador_6
}

#cargo audios y obtengo el tiempo de duración. Solo necesito buscar el de un solo microfono ya que las grabaciones se hicieron en simultaneo
nf_1_rec, nf_1_fs = auf.load_audio("2 - CASO 1 - AULA VACIA/2 - PISO DE RUIDO/MIC 1 - Earthworks M30_03.wav")
nf_2_rec, nf_2_fs = auf.load_audio("2 - CASO 1 - AULA VACIA/2 - PISO DE RUIDO/MIC 2 - Earthworks M30_03.wav")
nf_3_rec, nf_3_fs = auf.load_audio("2 - CASO 1 - AULA VACIA/2 - PISO DE RUIDO/MIC 3 - Behringer ecm 8000_05.wav")
nf_4_rec, nf_4_fs = auf.load_audio("2 - CASO 1 - AULA VACIA/2 - PISO DE RUIDO/MIC 4 - Behringer ecm 8000_04.wav")
nf_5_rec, nf_5_fs = auf.load_audio("2 - CASO 1 - AULA VACIA/2 - PISO DE RUIDO/MIC 5 - Behringer ecm 8000_04.wav")
nf_6_rec, nf_6_fs = auf.load_audio("2 - CASO 1 - AULA VACIA/2 - PISO DE RUIDO/MIC 6 - Behringer ecm 8000_04.wav")

#cargo valores de los sonómetros
thirds_sonometers_df, mr_sonometers_df = daf.load_sonometers_data("../datos sonometros/sonometros.xlsx")

thirds_bands_leq_1 = auf.leq_by_bands(nf_1_rec, thirds_filters_bank, ref=tono_calibrador_1)
thirds_bands_leq_6 = auf.leq_by_bands(nf_6_rec, thirds_filters_bank, ref=tono_calibrador_6)

thirds_bands_leq_s1 = daf.get_sonometer_leq(thirds_sonometers_df, nominal_thirds_central_freqs, "PR - S1 - SG")
thirds_bands_leq_s2 = daf.get_sonometer_leq(thirds_sonometers_df, nominal_thirds_central_freqs, "PR - S2 - SG")

#datos para graficar
mic1_thirds_leq_plot = {"leq":thirds_bands_leq_1, "label": "MIC 1"} 
s1_thirds_leq_plot = {"leq":thirds_bands_leq_s1, "label": "SONOMETRO 1"} 
mic6_thirds_leq_plot = {"leq":thirds_bands_leq_6, "label": "MIC 6"} 
s2_thirds_leq_plot = { "leq":thirds_bands_leq_s2, "label": "SONOMETRO 2"} 

titles={
    "1": "LEQ por 1/3 de octava entre microfono 1 y sonometro 1", 
    "2":"LEQ por 1/3 de octava entre microfono 1 y sonometro 1", 
    "3": "LEQ por 1/3 de octava entre microfono 1 y microfono 6", 
    "4": "LEQ por 1/3 de octava entre sonometros"
}

print(thirds_bands_leq_s1)
#plot_m1_s1_thirds = lambda: plot.plot_leqs(mic1_thirds_leq_plot, s1_thirds_leq_plot, x=nominal_thirds_central_freqs, title=titles["1"], show=False)
#plot_m6_s2_thirds = lambda: plot.plot_leqs(mic6_thirds_leq_plot, s2_thirds_leq_plot,  x=nominal_thirds_central_freqs, title=titles["2"], show=False)
#plot_m1_m6_thirds = lambda: plot.plot_leqs(mic1_thirds_leq_plot, mic6_thirds_leq_plot,  x=nominal_thirds_central_freqs, title=titles["3"], show=False)
#plot_s1_s2_thirds = lambda: plot.plot_leqs(s1_thirds_leq_plot, s2_thirds_leq_plot, x=nominal_thirds_central_freqs, title=titles["4"], show=False)

#plot.multiplot(plot_m1_s1_thirds, plot_m6_s2_thirds, plot_m1_m6_thirds, plot_s1_s2_thirds, figsize=(14, 10))
