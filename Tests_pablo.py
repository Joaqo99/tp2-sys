import numpy as np
import soundfile as sf
import plot as plot
import audio_functions as auf


##------------------------------------------------------------------------------------------------------------------------
import muestreo as ms
import pandas as pd   

#frecuencias para gráficar en tiempo y frecuencia.
frecuencia_1 = 125
frecuencia_2 = 1000
frecuencia_3 = 1500
#frecuencias para generar tono.
frecuencia_4 = 4000
frecuencia_5 = 12000
frecuencia_6 = 20000

duracion = 1
tabla_seno = pd.read_csv("sin_lookup.csv")

seno_1 = ms.gen_sin_table(duracion, frecuencia_1, tabla_seno)
t_2, seno_2 = ms.gen_sin_table(duracion, frecuencia_2, tabla_seno)
t_3, seno_3 = ms.gen_sin_table(duracion, frecuencia_3, tabla_seno)
t_4, seno_4 = ms.gen_sin_table(duracion, frecuencia_4, tabla_seno)
t_5, seno_5 = ms.gen_sin_table(duracion, frecuencia_5, tabla_seno)
t_6, seno_6 = ms.gen_sin_table(duracion, frecuencia_6, tabla_seno)
print(seno_1)

