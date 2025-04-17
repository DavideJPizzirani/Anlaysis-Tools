import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import AutoLocator, AutoMinorLocator

def resistivity_to_conductance(df_xx, df_xy, temperature, title1= "Sigma_xx", title2 = "Sigma_xy", xlabel = "Field (T)", ylabel1 = "Sigma_xx", ylabel2 = "Sigma_xy", show = False):
    sigma_xx = df_xx.copy()
    sigma_xx.y /= df_xx.y ** 2 + df_xy.y ** 2
    sigma_xy = df_xy.copy()
    sigma_xy.y /= df_xx.y ** 2 + df_xy.y ** 2
    if show is True:
        sigma_xx.plot(x="x", y="y", label = temperature)
        plt.title(title1)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel1)
        sigma_xy.plot(x="x", y="y", label = temperature)
        plt.title(title2)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel2)
    return (sigma_xx, sigma_xy)