import audio_functions as auf
import filters_bank as fb
import plot
import codigos_medicion as cm

#cargo audios y obtengo el tiempo de duración. Solo necesito buscar el de un solo microfono ya que las grabaciones se hicieron en simultaneo
nf_1_rec, nf_1_fs = auf.load_audio("audios/2 - CASO 1 - AULA VACIA/2 - PISO DE RUIDO/MIC 1 - Earthworks M30_03.wav")


nf_1_dur ,nf_1_time_array = auf.get_audio_time_array(nf_1_rec, nf_1_fs)
plot.plot_signal([nf_1_time_array, nf_1_rec])