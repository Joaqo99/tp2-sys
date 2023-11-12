import audio_functions as auf
import filters_bank as fb
import plot
import codigos_medicion as cm
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np
 
nominal_oct_central_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
exact_oct_central_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
nominal_thirds_central_freqs = [100, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0]
exact_thirds_central_freqs = [99.123, 125.0, 157.49, 198.43, 250.0, 314.98, 396.85, 500.0, 628.96, 793.7, 1000.0, 1259.9, 1587.4, 2000.0, 2519.8, 3174.8, 4000.0, 5039.7]

octave_filters_bank = fb.create_octaves_filter_bank(exact_oct_central_freqs, 48000, 3)
thirds_filters_bank = fb.create_thirds_filter_bank(exact_thirds_central_freqs, 48000, 4)

#busco los paths de las señales en el archivo de excel
cal_path = auf.get_paths("Paths.xlsx", "Calibracion")

#Entrega listas de los audios
calibraciones_fs, calibraciones = auf.get_signals(cal_path)

#cargo audios y obtengo el tiempo de duración. Solo necesito buscar el de un solo microfono ya que las grabaciones se hicieron en simultaneo
nf_sg_path = auf.get_paths("Paths.xlsx", "PR - SG")
nf_cg_path = auf.get_paths("Paths.xlsx", "PR - CG")

nf_sg_fs, nf_sg_recs = auf.get_signals(nf_sg_path)
nf_cg_fs, nf_cg_recs = auf.get_signals(nf_cg_path)

#calculo el leq para cada banda para todos los audios y los guardo en una lista para q ancho de banda

all_thirds_leq_sg = map(lambda x: auf.np.array(auf.leq_by_bands(x[0], thirds_filters_bank, x[1])), zip(nf_sg_recs, calibraciones))
all_octaves_leq_sg = map(lambda x: auf.np.array(auf.leq_by_bands(x[0], octave_filters_bank, x[1])), zip(nf_sg_recs, calibraciones))
all_thirds_leq_cg = map(lambda x: auf.np.array(auf.leq_by_bands(x[0], thirds_filters_bank, x[1])), zip(nf_cg_recs, calibraciones))
all_octaves_leq_cg = map(lambda x: auf.np.array(auf.leq_by_bands(x[0], octave_filters_bank, x[1])), zip(nf_cg_recs, calibraciones))

#paso a las listas a un numpy array para poder hacer el promedio entre bandas

print(all_thirds_leq_sg)

all_thirds_leq_sg = auf.np.array(all_thirds_leq_cg)
all_octaves_leq_sg = auf.np.array(all_octaves_leq_cg)
all_thirds_leq_cg = auf.np.array(all_thirds_leq_cg)
all_octaves_leq_cg = auf.np.array(all_octaves_leq_cg)

print(all_thirds_leq_sg)

avg_octave_leq_sg_nf = auf.np.mean(all_octaves_leq_sg, axis=0)
avg_thirds_leq_sg_nf = auf.np.mean(all_thirds_leq_sg, axis=0)
avg_octave_leq_cg_nf = auf.np.mean(all_octaves_leq_cg, axis=0)
avg_thirds_leq_cg_nf = auf.np.mean(all_thirds_leq_cg, axis=0)