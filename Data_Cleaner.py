#%%writefile nameofthescript.py
#%run nameofthescript (NO EXTENSION NEEDED)

#-------------------------#
    
#     GENERAL IMPORT
#           -
#       Libraries

#-------------------------#   
import os
import json
import pandas as pd
import numpy as np
import warnings
from numpy import float64
import scipy as py 
import math as ma
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as scintpl
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial import Polynomial
import ipywidgets as widgets
from ipywidgets import interact, fixed, widgets
from IPython.display import display
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import AutoLocator, AutoMinorLocator
import colorcet as cc
from typing import Any, Match, Optional, TextIO
from memspectrum import MESA

plt.style.use('dark_background')

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#--------------Data Cleaning---------------#

# Function removing all the rows containing any or all NaN, rows with all zeros and averaging zero field and max field y values.
# NOTE: alongside the zero field values, also !=0 duplicates in x will be binned as well.

'''	Function removing:
                        - all rows with NaN value;
                        - check if column x is monotonic, if not eliminate the unexpected point;
                        - rows with x = 0 and y = 0 (LABVIEW field value read out error).
 	:param df: pandas dataframe fro PvdB tool or any other file.
     return: pandas dataframe where x and y have been filtered and cleaned by nonsense values. '''

def remove_nan_zeros_bin_pandas(df):
    # remove all rows with NaN value, either in x or y, directly without creating a copy of df (inplace=True).
    df.dropna(axis=0, how="any", inplace=True)
    # reset index after NaN drop.
    df.reset_index(drop=True, inplace=True)
    # remove all rows with all zeros (e.g. x=0 and y=0).
    df=df.loc[(df!=0).any(axis=1)]
    # reset index after x=0 and y=0 drop.
    df.reset_index(drop=True, inplace=True)
    # check x is strictly monotonically increasing.
    while True:
        mon_inc = df['x'].diff().fillna(0) >= 0
        if mon_inc.all():
            break
        df = df[mon_inc]
    # reset index.
    df.reset_index(drop=True, inplace=True)
    # binning y values based on multiple values of magnetic field (like several 0s at the beginning and multiple max field values).
    df = df.groupby(['x']).mean().reset_index()
    # check if the column related to magnetic field values (B) has monotonically increasing and unique values.
    df['x'].is_monotonic_increasing and df['x'].is_unique
    # remove all y = 0
    # NOTE: if you enable this line then you assume that the crossover of Hall resistance will never hit, due to the loop time of the acquisition, y = 0.0.
    #       In general, you should NOT enable this line.
    #df = df[df.B != 0]
    # reset the index of the dataframe after drop
    #df.reset_index(drop=True, inplace=True)
    return df



def clean_dfs_in_list(list_to_be_cleaned, show = False):
    newlist = []
    for df in list_to_be_cleaned:
        newlist.append(remove_nan_zeros_bin_pandas(df))
        print(newlist)
        if show is True:
            for df in newlist:
                df.plot(x="x", y="y")
                plt.show()
    return newlist
# %%
