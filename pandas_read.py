import pandas as pd

def excel_sheets_data_to_DataFrame(file_name, *sheet_names):
    '''
    Reads excel' sheets by pandas like a DataFrame

    Inputs:
    file_name (srt):
        Excel file name to read
    sheet_names (srt):
        Sheet name or names from excel to take data
    ------------
    Output:
        data (DataFrame):
            DataFrame that conteins every data from sheets
    ------------
    Raices:
        TypeError: if file_name or sheet_names are not strings
    '''
    data = []
    for sheet in sheet_names:
        sheet = pd.read_excel(file_name, sheet_name = sheet)
        data.append(sheet)
    return data