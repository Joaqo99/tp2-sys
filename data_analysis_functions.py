import pandas as pd
import numpy as np

def load_sonometers_data(filename):
    """
    Opens an excel file 
    Input:
        - filename: str type object.
    Output:
        - thirds: DataFrame type object. Contains
    """

    thirds = pd.read_excel(filename, sheet_name="tercios")
    main_results = pd.read_excel(filename, sheet_name="main_results")
    return thirds, main_results

def get_sonometer_leq(data, central_freqs, *measurements):

    if len(measurements) == 1:
        i = measurements[0]
        leq_values = []
        for freq in data["Frequency [Hz]"].values:
            freq = int(np.rint(freq))
            if freq in central_freqs:
                filtro_frecuencias = data["Frequency [Hz]"] == freq
                leq = data.loc[filtro_frecuencias, i].values
                leq_values.append(float(leq))

        return leq_values

    elif len(measurements) > 1:
        all_leqs = []
        for i in measurements:
            leq_values = []
            for freq in data["Frequency [Hz]"].values:
                freq = int(np.rint(freq))
                if freq in central_freqs:
                    filtro_frecuencias = data["Frequency [Hz]"] == freq
                    leq = data.loc[filtro_frecuencias, i]
                    leq_values.append(leq)
            all_leqs.append(leq_values)    
        return all_leqs    