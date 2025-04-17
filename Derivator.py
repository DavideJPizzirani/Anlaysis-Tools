import pandas as pd
from pandas import DataFrame
import numpy as np
import math as ma
import matplotlib as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import AutoLocator, AutoMinorLocator
from scipy import signal
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as scintpl
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial import Polynomial
from typing import Any, Match, Optional, TextIO
import colorcet as cc
import plotly.graph_objects as go
from plotly.offline import iplot
import plotly.express as px

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

def convert_pd_dataframe_into_np_arrays_x(df):
    x_array = df.x.to_numpy()
    return x_array

def convert_pd_dataframe_into_np_arrays_y(df):
    y_array = df.y.to_numpy()
    return y_array

def interpolator_derivator(df, method, order, show = False, n_points: Optional[int] = 0, base2: bool = True, power: int = 0):
    # General assertion method on function variables
    assert type(method) is str
    assert type(order) is int
    assert type(n_points) is int
    assert type(power) is int
    x_array = convert_pd_dataframe_into_np_arrays_x(df)
    y_array = convert_pd_dataframe_into_np_arrays_y(df)
    if method != "user_defined" and method != "auto":
        print(bcolors.FAIL + "Invalid input. The only option available is : user_defined or auto." + bcolors.FAIL)
    if method == "auto" and n_points == 0 and order == 1:
        # linear interpolation with number of points of interpolation defined by smallest sampling interval available in x-data.
        # NOTE: linear interpolation is performed via scintpl.UnivariateSpline with smoothing factor set to 0.
        print(bcolors.OKGREEN + "You have selected linear interpolation with smallest sampling interval availabe in the x-data" + bcolors.ENDC)
        # get smallest sampling interval available in x-data
        dx: float = x_array[1] - x_array[0]
        # calculate the sampling interval used for interpolation based on the wanted number of points from original x-data
        n_points = int((x_array[-1] - x_array[0]) / dx) + 1
        if base2:
        # get next power of 2 if user did not specify any power to be used (power=0) OR
        # when a power has been given that would result in less points than based on the smallest sampling interval
            if not power or (2 ** power < n_points):
                # ceil is approximating to next nearest int
                # log provides the base 2 logarithm in order to evaluate the power, which will be approximated as next nearest int number
                power = ma.ceil(ma.log(n_points, 2))
                print(bcolors.UNDERLINE + "Next power of 2 is: " + str(power) + bcolors.ENDC)
                # calculate new total number of points for interpolation
            n_points = 2 ** power
        print(bcolors.UNDERLINE + "The total number of points for interpolation is " + str(n_points) + bcolors.ENDC)
        # calculate the sampling interval used for interpolation based on the wanted number of points from original x-data
        dx_intpl = (x_array[-1] - x_array[0]) / n_points
        # calculate the x-data values where the interpolator function will be evaluated at
        # NOTE: x[-1] will not be included in x_intpl
        x_intpl = np.arange(x_array[0], x_array[-1], dx_intpl)
        print(str(x_intpl.shape) + str(y_array.shape))
        interpolator = scintpl.UnivariateSpline(x_array, y_array, k=1, s=0)
        y_intpl = interpolator(x_intpl)
        #y_intpl = np.interp(x_intpl, x_array, y_array)
        intpl_signal = np.vstack((x_intpl, y_intpl)).T
        intpl_df = pd.DataFrame(intpl_signal, columns=['x', 'y'])
        print(intpl_df)
        first_derivative = interpolator.derivative()
        d1_intpl_signal = first_derivative(x_intpl)
        d1_signal = np.vstack((x_intpl, d1_intpl_signal)).T
        d1_df = pd. DataFrame(d1_signal, columns=['x', 'y'])
        if show is True:
            dfs = {"I derived signal": d1_df}
            fig = go.Figure()
            fig.update_layout(
            title="I derivative of longitudinal resistivity",
            xaxis_title= "Temperature (K)",
            yaxis_title="\u03C1\u02E3\u02E3' (\u03BC\u03A9 cm)",
            legend_title="I derivative signal",
            font=dict(
                family="Times New Roman, monospace",
                size=18,
                color="black"))
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
            for i in dfs:
                fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
            fig.show()
        return d1_df
    if method == "user_defined" and n_points != 0 and order == 1:
        # linear interpolation with number of points of interpolation defined by the user.
        # NOTE: linear interpolation is performed via scintpl.UnivariateSpline with smoothing factor set to 0.
        print(bcolors.OKBLUE + "You have selected linear interpolation with user defined number of points: " + str(n_points) + bcolors.ENDC)  
        dx_intpl = np.linspace(np.amin(x_array), np.amax(x_array), n_points)
        interpolator = scintpl.UnivariateSpline(x_array, y_array, k=1, s=0)
        y_intpl = interpolator(dx_intpl)
        #y_intpl = np.interp(dx_intpl, x_array, y_array)
        intpl_signal = np.vstack((dx_intpl, y_intpl)).T
        intpl_df = pd. DataFrame(intpl_signal, columns=['x', 'y'])
        print(intpl_df)
        first_derivative = interpolator.derivative()
        d1_intpl_signal = first_derivative(dx_intpl)
        d1_signal = np.vstack((dx_intpl, d1_intpl_signal)).T
        d1_df = pd. DataFrame(d1_signal, columns=['x', 'y'])
        if show is True:
            dfs = {"I derived signal": d1_df}
            fig = go.Figure()
            fig.update_layout(
            title="I derivative of longitudinal resistivity",
            xaxis_title= "Temperature (K)",
            yaxis_title="\u03C1\u02E3\u02E3' (\u03BC\u03A9 cm)",
            legend_title="I derivative signal",
            font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
            for i in dfs:
                fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
            fig.show()
        return d1_df
    if method == "auto" and n_points == 0 and order == 2:
        # linear interpolation with number of points of interpolation defined by smallest sampling interval available in x-data.
        # NOTE: linear interpolation is performed via scintpl.UnivariateSpline with smoothing factor set to 0.
        print(bcolors.OKGREEN + "You have selected linear interpolation with smallest sampling interval availabe in the x-data" + bcolors.ENDC)
        # get smallest sampling interval available in x-data
        dx: float = x_array[1] - x_array[0]
        # calculate the sampling interval used for interpolation based on the wanted number of points from original x-data
        n_points = int((x_array[-1] - x_array[0]) / dx) + 1
        if base2:
        # get next power of 2 if user did not specify any power to be used (power=0) OR
        # when a power has been given that would result in less points than based on the smallest sampling interval
            if not power or (2 ** power < n_points):
                # ceil is approximating to next nearest int
                # log provides the base 2 logarithm in order to evaluate the power, which will be approximated as next nearest int number
                power = ma.ceil(ma.log(n_points, 2))
                print(bcolors.UNDERLINE + "Next power of 2 is: " + str(power) + bcolors.ENDC)
                # calculate new total number of points for interpolation
            n_points = 2 ** power
        print(bcolors.UNDERLINE + "The total number of points for interpolation is " + str(n_points) + bcolors.ENDC)
        # calculate the sampling interval used for interpolation based on the wanted number of points from original x-data
        dx_intpl = (x_array[-1] - x_array[0]) / n_points
        # calculate the x-data values where the interpolator function will be evaluated at
        # NOTE: x[-1] will not be included in x_intpl
        x_intpl = np.arange(x_array[0], x_array[-1], dx_intpl)
        print(str(x_intpl.shape) + str(y_array.shape))
        interpolator = scintpl.UnivariateSpline(x_array, y_array, k=1, s=0)
        y_intpl = interpolator(x_intpl)
        #y_intpl = np.interp(x_intpl, x_array, y_array)
        intpl_signal = np.vstack((x_intpl, y_intpl)).T
        intpl_df = pd.DataFrame(intpl_signal, columns=['x', 'y'])
        print(intpl_df)
        first_derivative = interpolator.derivative()
        d1_intpl_signal = first_derivative(x_intpl)
        d1_signal = np.vstack((x_intpl, d1_intpl_signal)).T
        #2nd order derivative
        # Due to spline having order n=1 one has to interpolate again
        reinterpolation = scintpl.UnivariateSpline(x_intpl, d1_intpl_signal, k=1, s=0)
        second_derivative = reinterpolation.derivative()
        d2_intpl_signal = second_derivative(x_intpl)
        d2_signal = np.vstack((x_intpl, d2_intpl_signal)).T
        d2_df = pd. DataFrame(d2_signal, columns=['x', 'y'])
        if show is True:
            dfs = {"II derived signal": d2_df}
            fig = go.Figure()
            fig.update_layout(
            title="II derivative of longitudinal resistivity",
            xaxis_title= "Temperature (K)",
            yaxis_title="\u03C1\u02E3\u02E3'' (\u03BC\u03A9 cm)",
            legend_title="II derivative signal",
            font=dict(
                family="Times New Roman, monospace",
                size=18,
                color="black"))
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
            for i in dfs:
                fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
            fig.show()
        return d2_df
    if method == "user_defined" and n_points != 0 and order == 2:
        # linear interpolation with number of points of interpolation defined by the user.
        # NOTE: linear interpolation is performed via scintpl.UnivariateSpline with smoothing factor set to 0.
        print(bcolors.OKBLUE + "You have selected linear interpolation with user defined number of points: " + str(n_points) + bcolors.ENDC)  
        dx_intpl = np.linspace(np.amin(x_array), np.amax(x_array), n_points)
        interpolator = scintpl.UnivariateSpline(x_array, y_array, k=1, s=0)
        y_intpl = interpolator(dx_intpl)
        #y_intpl = np.interp(dx_intpl, x_array, y_array)
        intpl_signal = np.vstack((dx_intpl, y_intpl)).T
        intpl_df = pd. DataFrame(intpl_signal, columns=['x', 'y'])
        print(intpl_df)
        first_derivative = interpolator.derivative()
        d1_intpl_signal = first_derivative(dx_intpl)
        d1_signal = np.vstack((dx_intpl, d1_intpl_signal)).T
        #2nd order derivative
        # Due to spline having order n=1 one has to interpolate again
        reinterpolation = scintpl.UnivariateSpline(dx_intpl, d1_intpl_signal, k=1, s=0)
        second_derivative = reinterpolation.derivative()
        d2_intpl_signal = second_derivative(dx_intpl)
        d2_signal = np.vstack((dx_intpl, d2_intpl_signal)).T
        d2_df = pd. DataFrame(d2_signal, columns=['x', 'y'])
        if show is True:
            dfs = {"II derived signal": d2_df}
            fig = go.Figure()
            fig.update_layout(
            title="II derivative of longitudinal resistivity",
            xaxis_title= "Temperature (K)",
            yaxis_title="\u03C1\u02E3\u02E3'' (\u03BC\u03A9 cm)",
            legend_title="II derivative signal",
            font=dict(
            family="Times New Roman, monospace",
            size=18,
            color="black"))
            fig.update_xaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_xaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
            fig.update_yaxes(showline=True, linewidth=1, linecolor='black', mirror=True)
            fig.update_yaxes(ticks="inside", tickwidth=1, tickcolor='black', ticklen=10, mirror=True)
            for i in dfs:
                fig = fig.add_trace(go.Scatter(x = dfs[i].x, y = dfs[i].y, name = i, mode = 'markers'))
            fig.show()
        return d2_df
    else:
        print(bcolors.FAIL + "Wrong input parameters!" + bcolors.ENDC)