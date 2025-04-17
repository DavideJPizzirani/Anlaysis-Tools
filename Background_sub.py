import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy.polynomial import Polynomial
from numpy.polynomial.polynomial import polyfit
#----------------------------------------------------------------#
    
#                       BACKGROUND REMOVAL

#----------------------------------------------------------------#


def convert_pd_dataframe_into_np_arrays_x(df):
    x_array = df.x.to_numpy()
    return x_array

def convert_pd_dataframe_into_np_arrays_y(df):
    y_array = df.y.to_numpy()
    return y_array

#--------------BACKGROUND DEFINING FUNCTION---------------#

def fit_background(x_array, y_array, kind):
    assert type(kind) is int
    assert len(x_array) > 0
    assert len(y_array) > 0
    poly = Polynomial(polyfit(x_array, y_array, deg = kind))
    return poly

#--------------BACKGROUND REMOVED CURVE IN B IN AN USER DEFINED RANGE ---------------#

def remove_bg(x_array, y_array, x_start, x_end, order):
    assert type(order) is int
    if type(x_start) is str and type(x_end) is str:
        maximum = np.amax(x_array)
        minimum = np.amin(x_array[1:])
        bool_idx = (x_array >= minimum) * (x_array <= maximum)
        # return the data points falling into selected range by using the boolean array as boolean index
        x_array = x_array[bool_idx]
        y_array = y_array[bool_idx]
        poly = fit_background(x_array, y_array, kind = order)
        # assign the two arrays 
        array_no_bg_x = x_array
        array_no_bg_y = y_array
        # perform the background subtraction
        array_no_bg_y -= poly(array_no_bg_x)
        # merge the 1D np.ndarray to a 2D np.ndarray
        no_bg_array = np.vstack((array_no_bg_x,  array_no_bg_y)).T
    if type(x_start) is float and type(x_end) is float: # define boolean array with the same length as x
        bool_idx = (x_array >= x_start) * (x_array <= x_end)
        # return the data points falling into selected range by using the boolean array as boolean index
        x_array = x_array[bool_idx]
        y_array = y_array[bool_idx]
        #mask = (x_array>x_start) & (x_array<x_end)
        poly = fit_background(x_array, y_array, kind = order)
        # assign the two arrays 
        array_no_bg_x = x_array
        array_no_bg_y = y_array
        # perform the background subtraction
        array_no_bg_y -= poly(array_no_bg_x)
        # merge the 1D np.ndarray to a 2D np.ndarray
        no_bg_array = np.vstack((array_no_bg_x,  array_no_bg_y)).T
    return no_bg_array


#--------------BACKGROUND REMOVED CURVE & TRANSLATE IN 1/B IN AN USER DEFINED RANGE ---------------#

def remove_bg_invert_x(x_array, y_array, x_start, x_end, order):
    assert type(order) is int
    assert x_end != 0
    assert x_start != 0
    if type(x_start) is str and type(x_end) is str:
        maximum = np.amax(x_array)
        minimum = np.amin(x_array[1:])
        bool_idx = (x_array >= minimum) * (x_array <= maximum)
        # return the data points falling into selected range by using the boolean array as boolean index
        x_array = x_array[bool_idx]
        y_array = y_array[bool_idx]
        poly = fit_background(x_array, y_array, kind = order)
        # assign the two arrays 
        array_no_bg_x = x_array
        array_no_bg_y = y_array
        # perform the background subtraction
        array_no_bg_y -= poly(array_no_bg_x)
        # invert x values (from B- to 1/B - domain) in the selected range
        array_no_bg_x = 1 / array_no_bg_x
        # merge the 1D np.ndarray to a 2D np.ndarray
        no_bg_array = np.vstack((array_no_bg_x,  array_no_bg_y)).T
        return no_bg_array
    if type(x_start) is float and type(x_end) is float: 
        # define boolean array with the same length as x
        bool_idx = (x_array >= x_start) * (x_array <= x_end)
        # return the data points falling into selected range by using the boolean array as boolean index
        x_array = x_array[bool_idx]
        y_array = y_array[bool_idx]
        #mask = (x_array>x_start) & (x_array<x_end)
        poly = fit_background(x_array, y_array, kind = order)
        # assign the two arrays 
        array_no_bg_x = x_array
        array_no_bg_y = y_array
        # perform the background subtraction
        array_no_bg_y -= poly(array_no_bg_x)
        # invert x values (from B- to 1/B - domain) in the selected range
        array_no_bg_x = 1 / array_no_bg_x
        # merge the 1D np.ndarray to a 2D np.ndarray
        no_bg_array = np.vstack((array_no_bg_x, array_no_bg_y)).T
        return no_bg_array



#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#--------------BACKGROUND REMOVED CURVE IN B IN AN USER DEFINED RANGE ---------------#


# function converting the panda dataframes in a list into np arrays

def convert_pd_dataframe_into_np_arrays_multiple(list):
    new_list = []
    for df in list:
        array = df.to_numpy()
        new_list.append(array)
        return new_list

#------------------------------------------------------------------------------------#

def remove_bg_df(dfs, x_start, x_end, order):
    assert type(order) is int
    if type(x_start) is str and type(x_end) is str:
        no_bg_list = []
        for df in dfs:
            x_array = convert_pd_dataframe_into_np_arrays_x(df)
            y_array = convert_pd_dataframe_into_np_arrays_y(df)
            maximum = np.amax(x_array)
            minimum = np.amin(x_array)
            bool_idx = (x_array >= minimum) * (x_array <= maximum)
            # return the data points falling into selected range by using the boolean array as boolean index
            x_array = x_array[bool_idx]
            y_array = y_array[bool_idx]
            poly = fit_background(x_array, y_array, kind = order)
            # assign the two arrays 
            array_no_bg_x = x_array
            array_no_bg_y = y_array
            # perform the background subtraction
            array_no_bg_y -= poly(array_no_bg_x)
            # merge the 1D np.ndarray to a 2D np.ndarray
            no_bg_array = np.vstack((array_no_bg_x,  array_no_bg_y)).T
            df_no_bg = pd.DataFrame(no_bg_array, columns=['x', 'y'])
            no_bg_list.append(df_no_bg)
        return no_bg_list
    if type(x_start) is float and type(x_end) is float: # define boolean array with the same length as x
        no_bg_list = []
        for df in dfs:
            x_array = convert_pd_dataframe_into_np_arrays_x(df)
            y_array = convert_pd_dataframe_into_np_arrays_y(df)
            bool_idx = (x_array >= x_start) * (x_array <= x_end)
            # return the data points falling into selected range by using the boolean array as boolean index
            x_array = x_array[bool_idx]
            y_array = y_array[bool_idx]
            #mask = (x_array>x_start) & (x_array<x_end)
            poly = fit_background(x_array, y_array, kind = order)
            # assign the two arrays 
            array_no_bg_x = x_array
            array_no_bg_y = y_array
            # perform the background subtraction
            array_no_bg_y -= poly(array_no_bg_x)
            # merge the 1D np.ndarray to a 2D np.ndarray
            no_bg_array = np.vstack((array_no_bg_x,  array_no_bg_y)).T
            df_no_bg = pd.DataFrame(no_bg_array, columns=['x', 'y'])
            no_bg_list.append(df_no_bg)
        return no_bg_list


#--------------BACKGROUND REMOVED CURVE & TRANSLATE IN 1/B IN AN USER DEFINED RANGE ---------------#

def remove_bg_invert_x_multiple(dfs, x_start, x_end, order):
    assert type(order) is int
    assert x_end != 0
    assert x_start != 0
    if type(x_start) is str and type(x_end) is str:
        no_bg_list = []
        for df in dfs:
            x_array = convert_pd_dataframe_into_np_arrays_x(df)
            y_array = convert_pd_dataframe_into_np_arrays_y(df)
            maximum = np.amax(x_array)
            minimum = np.amin(x_array[1:])
            bool_idx = (x_array >= minimum) * (x_array <= maximum)
            # return the data points falling into selected range by using the boolean array as boolean index
            x_array = x_array[bool_idx]
            y_array = y_array[bool_idx]
            poly = fit_background(x_array, y_array, kind = order)
            # assign the two arrays 
            array_no_bg_x = x_array
            array_no_bg_y = y_array
            # perform the background subtraction
            array_no_bg_y -= poly(array_no_bg_x)
            # invert x values (from B- to 1/B - domain) in the selected range
            array_no_bg_x = 1 / array_no_bg_x
            # merge the 1D np.ndarray to a 2D np.ndarray
            no_bg_array = np.vstack((array_no_bg_x,  array_no_bg_y)).T
            df_no_bg = pd.DataFrame(no_bg_array, columns=['x', 'y'])
            no_bg_list.append(df_no_bg)
        return no_bg_list
    if type(x_start) is float and type(x_end) is float: 
        # define boolean array with the same length as x
        no_bg_list = []
        for df in dfs:
            x_array = convert_pd_dataframe_into_np_arrays_x(df)
            y_array = convert_pd_dataframe_into_np_arrays_y(df)
            bool_idx = (x_array >= x_start) * (x_array <= x_end)
            # return the data points falling into selected range by using the boolean array as boolean index
            x_array = x_array[bool_idx]
            y_array = y_array[bool_idx]
            #mask = (x_array>x_start) & (x_array<x_end)
            poly = fit_background(x_array, y_array, kind = order)
            # assign the two arrays 
            array_no_bg_x = x_array
            array_no_bg_y = y_array
            # perform the background subtraction
            array_no_bg_y -= poly(array_no_bg_x)
            # invert x values (from B- to 1/B - domain) in the selected range
            array_no_bg_x = 1 / array_no_bg_x
            # merge the 1D np.ndarray to a 2D np.ndarray
            no_bg_array = np.vstack((array_no_bg_x, array_no_bg_y)).T
            df_no_bg = pd.DataFrame(no_bg_array, columns=['x', 'y'])
            no_bg_list.append(df_no_bg)
        return no_bg_list