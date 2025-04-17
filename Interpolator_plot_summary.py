import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as scintpl
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial import Polynomial

def convert_pd_dataframe_into_np_arrays_x(df):
    x_array = df.x.to_numpy()
    return x_array

def convert_pd_dataframe_into_np_arrays_y(df):
    y_array = df.y.to_numpy()
    return y_array

def user_defined_interpolation_smoothing_derivative_Tsweep(list, num, show = True, save = False):
    assert type(num) is int
    #assert type(title) is str
    #assert type(xlabel) is str
    #assert type(ylabel) is str
    int_df = []
    Ider_df = []
    IIder_df = []
    for df in list:
        print(df.count())
        if len(df.index) < num:
            element_in_df = df.count()
            print("Number of points in df is: " + str(element_in_df))
            print("Number of points used for interpolation: " + str(num) + " is accepted.")
            x = convert_pd_dataframe_into_np_arrays_x(df)
            y = convert_pd_dataframe_into_np_arrays_y(df)
            x_intpl = np.linspace(np.amin(x), np.amax(x), num)
            y_intpl = np.interp(x_intpl, x, y)
            intpl_signal = np.vstack((x_intpl, y_intpl)).T
            #x = np.linspace(df.x.min(), df.x.max(), num)
            #y = np.interp(x, df.x, df.y)
            #int_df.append(pd.DataFrame({"x": x, "y": y}))
            d1_intpl_signal = np.gradient(y_intpl,x_intpl)
            d1_signal = np.vstack((x_intpl, d1_intpl_signal)).T
            reinterpolation =  np.interp(x_intpl, x_intpl, d1_intpl_signal)
            d2_intpl_signal = np.gradient(reinterpolation, x_intpl)
            d2_signal = np.vstack((x_intpl, d2_intpl_signal)).T
            intpl_df_signal = pd.DataFrame(intpl_signal, columns = ['x','y'])
            d1_df_signal = pd.DataFrame(d1_signal, columns = ['x','y'])
            d2_df_signal = pd.DataFrame(d2_signal, columns = ['x','y'])
            int_df.append(intpl_df_signal)
            Ider_df.append(d1_df_signal)
            IIder_df.append(d2_df_signal)
        else:
            raise ValueError
        if save:
            intpl_df_signal.to_csv("intpl_df_signal.csv", sep='\t', encoding='utf-8')
            d1_df_signal.to_csv("d1_df_signal.csv", sep='\t', encoding='utf-8')
            d2_df_signal.to_csv("d2_df_signal.csv", sep='\t', encoding='utf-8')
        else:
            pass
    if show is True:
        fig, axs = plt.subplots(2, 2)
        for df in list:
            axs[0, 0].plot(x, y, '+', color = "red" )
            axs[0, 0].set_title("Raw data", fontsize=15)
            axs[0, 0].set_xlabel("T(K)", fontsize=12, labelpad=12)
            axs[0, 0].set_ylabel("Signal", fontsize=12, labelpad=12)
        for df in int_df:
            axs[1, 0].plot(*zip(*intpl_signal), '+', color = "white")
            axs[1, 0].set_title("Interpolated data", fontsize=15)
            axs[1, 0].sharex(axs[0, 0])
            axs[1, 0].set_xlabel("T(K)", fontsize=12, labelpad=12)
            axs[1, 0].set_ylabel("Interpolated signal", fontsize=12, labelpad=12)
        for df in Ider_df:
            axs[0, 1].plot(*zip(*d1_signal), '+', color = "blue")
            axs[0, 1].set_title("Ist derivative", fontsize=15)
            axs[0, 1].set_xlabel("T(K)", fontsize=12, labelpad=12)
            axs[0, 1].set_ylabel("Ist derived signal", fontsize=12, labelpad=12)
        for df in IIder_df:
            axs[1, 1].plot(*zip(*d2_signal), '+', color = "green")
            axs[1, 1].set_title("IInd derivative", fontsize=15)
            axs[1, 1].sharex(axs[0, 1])
            axs[1, 1].set_xlabel("T(K)", fontsize=12, labelpad=12)
            axs[1, 1].set_ylabel("IIst derived signal", fontsize=12, labelpad=12)
        fig.tight_layout()
        plt.show()
    return int_df, Ider_df, IIder_df