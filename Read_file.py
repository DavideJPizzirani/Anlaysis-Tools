

#----------------------------------------------------------------#
    
#                   FUNCTION TO READ THE FILE

#----------------------------------------------------------------#

import pandas as pd
from pandas import DataFrame

def read_dat_file(file, separator, rows_to_skip, header, columns, labels, nature):
    df = pd.read_csv(file, sep = separator, skiprows=rows_to_skip, header = header,  usecols = columns, names = labels, dtype = nature, engine='python')
    return df
