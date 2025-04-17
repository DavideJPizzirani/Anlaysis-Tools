import pandas as pd
from pandas import DataFrame
import numpy as np


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

#----------------------------------------------------------------#
    
#          GET 1/B x - VALUES & CHECK DATA SET BOUNDARIES 

#----------------------------------------------------------------#


"""
    Function that creates a tuple containing the boundaries of an inverse magnetic field range.
    The function checks and returns the lowest and highest boundary for the data set based on the choiche of the user. 
    The function checks whether the MIN and MAX values are actually present in the np.ndarray. If not they are set automatically
    to the ones of the data set, MIN_dataset and MAX_dataset. 
    :param x_array : numpy.ndarray containing the values in B domain.
    :param MIN: smaller magnetic field value (B-space).
    :param MAX: larger magnetic field value (B-space).
    :return: MIN and MAX in 1/B-space domain, stored in invMIN and invMAX."""

def inverse_field_range(x_array, MIN, MAX):
    assert len(x_array) > 0
    assert type(MIN) is float
    assert MIN != 0
    assert type(MAX) is float
    MAX_dataset = np.amax(x_array)
    MIN_dataset = np.amin(x_array[1:])
    # Print lowest and highest value of the np.array
    print(bcolors.UNDERLINE + "Lowest boundary is: " + str(MIN_dataset) + " T" + " and highest boundary is: " + str(MAX_dataset)+ " T " + bcolors.ENDC)
    # Check if MIN and MAX exists in the x np.ndarray
    invMAX = 1 / MIN
    invMIN = 1 / MAX
    if MIN_dataset < MIN < MAX_dataset and MIN_dataset < MAX < MAX_dataset:
        print(bcolors.OKGREEN + "You choose B_min: " + str(MIN) + " T and B_max: " + str(MAX) + " T\n meaning 1/B_min: " + str(invMIN) + " 1/T" +  " and 1/B_max: " + str(invMAX) + " 1/T"  + bcolors.ENDC)
        return invMIN, invMAX
    if not MIN_dataset < MIN < MAX_dataset or MIN == 0.0 and MIN_dataset < MAX < MAX_dataset:
        #invMIN = 1 / MAX
        invMAX = 1 / MIN_dataset
        print(bcolors.FAIL + "MIN not contained in the array (or you choose 0, but 1/0 is not defined), reset user range to MIN_dataset!" + bcolors.ENDC)
        return invMIN, invMAX
    if not MIN_dataset < MAX < MAX_dataset and MIN_dataset < MIN < MAX_dataset:
        #invMAX = 1 / MIN
        invMIN = 1 / MAX_dataset
        print(bcolors.FAIL + "MAX not contained in the array, reset user range to MAX_dataset!" + bcolors.ENDC)
        return invMIN, invMAX  
    else: 
        print(bcolors.WARNING + "MIN and MAX not contained in the array, please check and enter valid boundaries!" + bcolors.ENDC)


#----------------------------------------------------------------#
    
#   CONVERT B DOMAIN TO 1/B DOMAIN AND RELATED y-VALUES BASED ON 
#         USER-DEFINED (AND PREVIOUSLY CHECKED) INTERVAL 
 

#----------------------------------------------------------------#

"""
    Function that creates a 2D np.ndarray containing a monotonically increasing 1/B converted values and the related y-value and related y-values based on a range (in B) 
    selected by the user.
    :param x_array : numpy.ndarray containing the values in B domain.
    :param y_array : numpy.ndarray containing the y-data. 
    :param min_bound: float indicating the lower boundary of the range
    :param max_bound: float indicating the upper boundary of the range
    :return: tuple of 2 numpy.ndarrays containing reversed 1/B- and related y-data within the range specified by the user"""

def convert_B_to_invB(x_array, y_array, min_bound, max_bound):
    assert len(x_array) > 0
    assert len(y_array) > 0
    assert type(min_bound) is float or type(None)
    assert type(max_bound) is float or type(None)
    assert min_bound != 0
    assert max_bound != 0
    # exclude (x=0, y) data point since x=0 is not defined
    if not x_array[0] and min_bound is None and max_bound is None:
        print(bcolors.OKBLUE + "You choose to compute over the full B range"  + bcolors.ENDC)
        # compute on the full B range
        x_array = x_array[1:]
        y_array = y_array[1:]
        xinv = 1/x_array
        # make x- and y-related values monotonic
        xinv = xinv[::-1]
        y_array = y_array[::-1]
        # merge the inverted x and y 1 D np.ndarray in on 2D np.ndarray
        inv_array = np.vstack((xinv, y_array)).T
    if not x_array[0] and min_bound is not None and max_bound is None:
        # compute from a specific B value
        invmax_bound = 1/min_bound
        #invmin_bound = 1/max_bound
        print(bcolors.OKBLUE + "You choose B_min: " + str(min_bound) + " T and B_max: " + "maximum value" + " T\n meaning 1/B_min: " + "from inverse of maximum value in " + " 1/T" +  " and 1/B_max: " + str(invmax_bound) + " 1/T"  + bcolors.ENDC)
        x_array = x_array[1:]
        y_array = y_array[1:]
        xinv = 1/x_array
        # make x- and y-related values monotonic
        xinv = xinv[::-1]
        y_array = y_array[::-1]
        # define boolean array with the same length as x
        bool_idx = (xinv <= invmax_bound)
        # return the data points falling into selected range by using the boolean array as boolean index
        xinv = xinv[bool_idx]
        y_array = y_array[bool_idx]
        # merge the inverted x and y 1D np.ndarray into a 2D np.ndarray
        inv_array = np.vstack((xinv, y_array)).T
    if not x_array[0] and min_bound is None and max_bound is not None:
        # compute up to a specific B value
        #invmax_bound = 1/min_bound
        invmin_bound = 1/max_bound
        print(bcolors.OKBLUE + "You choose B_min: " + "minimum value" + " T and B_max: " + str(max_bound) + " T\n meaning 1/B_min: " + str(invmin_bound) + " 1/T" +  " and max value in 1/T"  + bcolors.ENDC)
        x_array = x_array[1:]
        y_array = y_array[1:]
        xinv = 1/x_array
        # make x- and y-related values monotonic
        xinv = xinv[::-1]
        y_array = y_array[::-1]
        # define boolean array with the same length as x
        bool_idx = (xinv >= invmin_bound)
        # return the data points falling into selected range by using the boolean array as boolean index
        xinv = xinv[bool_idx]
        y_array = y_array[bool_idx]
        # merge the inverted x and y 1D np.ndarray into a 2D np.ndarray
        inv_array = np.vstack((xinv, y_array)).T
    if not x_array[0] and min_bound is not None and max_bound is not None:
        # compute on a specific B range
        invmax_bound = 1/min_bound
        invmin_bound = 1/max_bound
        print(bcolors.OKBLUE + "You choose B_min: " + str(min_bound) + " T and B_max: " + str(max_bound) + " T\n meaning 1/B_min: " + str(invmax_bound) + " 1/T" +  " and 1/B_max: " + str(invmin_bound) + " 1/T"  + bcolors.ENDC)
        x_array = x_array[1:]
        y_array = y_array[1:]
        xinv = 1/x_array
        # make x- and y-related values monotonic
        xinv = xinv[::-1]
        y_array = y_array[::-1]
        # define boolean array with the same length as x
        bool_idx = (xinv >= invmin_bound) * (xinv <= invmax_bound)
        # return the data points falling into selected range by using the boolean array as boolean index
        xinv = xinv[bool_idx]
        y_array = y_array[bool_idx]
        # merge the inverted x and y 1D np.ndarray into a 2D np.ndarray
        inv_array = np.vstack((xinv, y_array)).T
    return inv_array








#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#


def inverse_field_range_multiple(dfs, MIN, MAX):
    #assert len(x_array) > 0
    assert type(MIN) is float
    assert MIN != 0
    assert type(MAX) is float
    for df in dfs:
        x_array = convert_pd_dataframe_into_np_arrays_x(df)
        MAX_dataset = np.amax(x_array)
        MIN_dataset = np.amin(x_array[1:])
        # Print lowest and highest value of the np.array
        print(bcolors.UNDERLINE + "Lowest boundary is: " + str(MIN_dataset) + " T" + " and highest boundary is: " + str(MAX_dataset)+ " T " + bcolors.ENDC)
        # Check if MIN and MAX exists in the x np.ndarray
        invMAX = 1 / MIN
        invMIN = 1 / MAX
        if MIN_dataset < MIN < MAX_dataset and MIN_dataset < MAX < MAX_dataset:
            print(bcolors.OKGREEN + "You choose B_min: " + str(MIN) + " T and B_max: " + str(MAX) + " T\n meaning 1/B_min: " + str(invMIN) + " 1/T" +  " and 1/B_max: " + str(invMAX) + " 1/T"  + bcolors.ENDC)
            return invMIN, invMAX
        if not MIN_dataset < MIN < MAX_dataset or MIN == 0.0 and MIN_dataset < MAX < MAX_dataset:
            #invMIN = 1 / MAX
            invMAX = 1 / MIN_dataset
            print(bcolors.FAIL + "MIN not contained in the array (or you choose 0, but 1/0 is not defined), reset user range to MIN_dataset!" + bcolors.ENDC)
            return invMIN, invMAX
        if not MIN_dataset < MAX < MAX_dataset and MIN_dataset < MIN < MAX_dataset:
            #invMAX = 1 / MIN
            invMIN = 1 / MAX_dataset
            print(bcolors.FAIL + "MAX not contained in the array, reset user range to MAX_dataset!" + bcolors.ENDC)
            return invMIN, invMAX  
        else: 
            print(bcolors.WARNING + "MIN and MAX not contained in the array, please check and enter valid boundaries!" + bcolors.ENDC)


#----------------------------------------------------------------#
    
#   CONVERT B DOMAIN TO 1/B DOMAIN AND RELATED y-VALUES BASED ON 
#         USER-DEFINED (AND PREVIOUSLY CHECKED) INTERVAL 
 

#----------------------------------------------------------------#


def convert_B_to_invB_multiple(dfs, min_bound, max_bound):
    assert type(min_bound) is float or type(None)
    assert type(max_bound) is float or type(None)
    assert min_bound != 0
    assert max_bound != 0
    # exclude (x=0, y) data point since x=0 is not defined
    inv_b_list = []
    for df in dfs:
        x_array = convert_pd_dataframe_into_np_arrays_x(df)
        y_array = convert_pd_dataframe_into_np_arrays_y(df)
        if not x_array[0] and min_bound is None and max_bound is None:
            print(bcolors.OKBLUE + "You choose to compute over the full B range"  + bcolors.ENDC)
            # compute on the full B range
            x_array = x_array[1:]
            y_array = y_array[1:]
            xinv = 1/x_array
            # make x- and y-related values monotonic
            xinv = xinv[::-1]
            y_array = y_array[::-1]
            # merge the inverted x and y 1 D np.ndarray in on 2D np.ndarray
            inv_array = np.vstack((xinv, y_array)).T
            inv_b_list.append(inv_array)
            return inv_b_list
        if not x_array[0] and min_bound is not None and max_bound is None:
            # compute from a specific B value
            invmax_bound = 1/min_bound
            #invmin_bound = 1/max_bound
            print(bcolors.OKBLUE + "You choose B_min: " + str(min_bound) + " T and B_max: " + "maximum value" + " T\n meaning 1/B_min: " + "from inverse of maximum value in " + " 1/T" +  " and 1/B_max: " + str(invmax_bound) + " 1/T"  + bcolors.ENDC)
            x_array = x_array[1:]
            y_array = y_array[1:]
            xinv = 1/x_array
            # make x- and y-related values monotonic
            xinv = xinv[::-1]
            y_array = y_array[::-1]
            # define boolean array with the same length as x
            bool_idx = (xinv <= invmax_bound)
            # return the data points falling into selected range by using the boolean array as boolean index
            xinv = xinv[bool_idx]
            y_array = y_array[bool_idx]
            # merge the inverted x and y 1D np.ndarray into a 2D np.ndarray
            inv_array = np.vstack((xinv, y_array)).T
            inv_b_list.append(inv_array)
            return inv_b_list
        if not x_array[0] and min_bound is None and max_bound is not None:
            # compute up to a specific B value
            #invmax_bound = 1/min_bound
            invmin_bound = 1/max_bound
            print(bcolors.OKBLUE + "You choose B_min: " + "minimum value" + " T and B_max: " + str(max_bound) + " T\n meaning 1/B_min: " + str(invmin_bound) + " 1/T" +  " and max value in 1/T"  + bcolors.ENDC)
            x_array = x_array[1:]
            y_array = y_array[1:]
            xinv = 1/x_array
            # make x- and y-related values monotonic
            xinv = xinv[::-1]
            y_array = y_array[::-1]
            # define boolean array with the same length as x
            bool_idx = (xinv >= invmin_bound)
            # return the data points falling into selected range by using the boolean array as boolean index
            xinv = xinv[bool_idx]
            y_array = y_array[bool_idx]
            # merge the inverted x and y 1D np.ndarray into a 2D np.ndarray
            inv_array = np.vstack((xinv, y_array)).T
            inv_b_list.append(inv_array)
            return inv_b_list
        if not x_array[0] and min_bound is not None and max_bound is not None:
            # compute on a specific B range
            invmax_bound = 1/min_bound
            invmin_bound = 1/max_bound
            print(bcolors.OKBLUE + "You choose B_min: " + str(min_bound) + " T and B_max: " + str(max_bound) + " T\n meaning 1/B_min: " + str(invmax_bound) + " 1/T" +  " and 1/B_max: " + str(invmin_bound) + " 1/T"  + bcolors.ENDC)
            x_array = x_array[1:]
            y_array = y_array[1:]
            xinv = 1/x_array
            # make x- and y-related values monotonic
            xinv = xinv[::-1]
            y_array = y_array[::-1]
            # define boolean array with the same length as x
            bool_idx = (xinv >= invmin_bound) * (xinv <= invmax_bound)
            # return the data points falling into selected range by using the boolean array as boolean index
            xinv = xinv[bool_idx]
            y_array = y_array[bool_idx]
            # merge the inverted x and y 1D np.ndarray into a 2D np.ndarray
            inv_array = np.vstack((xinv, y_array)).T
            inv_b_list.append(inv_array)
            return inv_b_list
