import audio_functions as auf
import filters_bank as fb
import plot
import data_analysis_functions as daf
import numpy as np
import matplotlib.pyplot as plt

#nominal_oct_central_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
#exact_oct_central_freqs = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0]
#nominal_thirds_central_freqs = [100, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0, 500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0]
#exact_thirds_central_freqs = [99.123, 125.0, 157.49, 198.43, 250.0, 314.98, 396.85, 500.0, 628.96, 793.7, 1000.0, 1259.9, 1587.4, 2000.0, 2519.8, 3174.8, 4000.0, 5039.7]
#
#thirds_sonometers_df, mr_sonometers_df = daf.load_sonometers_data("../datos sonometros/sonometros.xlsx")

t_1 = np.linspace(0, 4, endpoint=True)
sin_1 = np.sin(2*np.pi*t_1)
t_2 = t_1*1/2
plt.plot(t_1, sin_1)
plt.show()

plt.plot(t_1, sin_1)
plt.plot(t_2, sin_1)
plt.show()