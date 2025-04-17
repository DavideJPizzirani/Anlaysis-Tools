
#----------------------------------------------------------------#
    
#                CONVERT DATAFRAME TO 2D NUMPY ARRAYS

#----------------------------------------------------------------#

# function converting the panda dataframe into np arrays

def convert_pd_dataframe_into_np_arrays(df):
    #x_array = df.x.to_numpy()
    #y_array = df.y.to_numpy()
    array = df.to_numpy()
    return array
    #return x_array, y_array

#----------------------------------------------------------------#
    
#                CONVERT df.x TO 1D np.array

#----------------------------------------------------------------#

# function converting the panda dataframe into np arrays

def convert_pd_dataframe_into_np_arrays_x(df):
    x_array = df.x.to_numpy()
    return x_array

#----------------------------------------------------------------#
    
#                CONVERT df.y TO 1D np.array

#----------------------------------------------------------------#

# function converting the panda dataframe into np arrays

def convert_pd_dataframe_into_np_arrays_y(df):
    y_array = df.y.to_numpy()
    return y_array
