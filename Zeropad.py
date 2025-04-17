
from typing import Any, Match, Optional, TextIO


#----------------------------------------------------------------#
    
#                           ZERO PADDING

#----------------------------------------------------------------#


"""
    function that zero pads y-data.

    :param y_array: np.ndarray of input y-data
    :param method: str indicating whether the function will work on auto-mode or user-defined mode
    :param base2: bool indicating that the amount of points is based on power of base 2
    :param power_base2: int giving the exponent of the power of base 2
    :param n_total: integer indicating the total number of points the data has to have after zero padding
    :return: np.ndarray of zero padded y-data"""

def zeropadding(y_array, method, base2 = True, power_base2 = 0, n_total = Optional[int]):
    assert len(y_array) > 0
    assert type(method) is str and method in ["auto", "user_defined"]
    assert power_base2 is int
    #case user doesn't want to zero pad, i.e. n_total = 0
    if n_total == 0:
        # set equal to length of y-data because no 0s are added after the last data point
        print(bcolors.OKGREEN + "You have selected not to zero pad the data. Number of zeros added after the last datapoint is:" + str(0) + bcolors.ENDC)
        n_total = len(y)
        print(bcolors.OKGREEN + "Total number of points coincide with lenght of the y np.ndarray:" + str(len(y)) + bcolors.ENDC)
    if method == "user_defined" and power_base2 != 0:
        print(bcolors.OKGREEN + "You have selected user-defined mode. The total number of points is decided upon input of power of base 2 from the user, which is:" + str(power_base2) + bcolors.ENDC)
        n_total = 2**power_base2
        print(bcolors.OKGREEN + "This corresponds to a total number of points of: " + str(n_total) + bcolors.ENDC)
        #calculate the actual zeros that will be added after the last datapoint of the np.ndarray
        number_of_zeros_after = n_total - len(y_array)
        print(bcolors.OKGREEN + "This corresponds to a total number of zeros added after the last datapoint of: " + str(n_total) + bcolors.ENDC)
        # perform and return zero padding of y-data
        return np.pad(y_array, number_of_zeros_after)
    if method == "auto" and power_base2 == 0:
        print(bcolors.OKGREEN + "You have selected auto-mode. The (next) power of 2 will mbe automatically calculated wrt to the initial length of array to be padded." + bcolors.ENDC)
        #automatically use the power of the next power of 2 (wrt N) if no external input is given (i.e. power_base2=0)
        power_base2 = ma.ceil(ma.log(len(y_array), 2)) + 1
        #convert power given in float into int
        power_base2 = ma.ceil(power_base2)
        print(bcolors.OKGREEN + "The power of 2 based on the length of the np.ndarray is:" + str(power_base2) + bcolors.ENDC)
        #check that the total number of resulting points based on the automatically calculated power of 2 is larger than length of y npndarray.
    while power_base2 <= ma.ceil(ma.log(len(y_array), 2)):
            print(f'power={power_base2} for total number of points with zeros padded smaller than/equal to length of'
                  'y-data!')
            power_base2 += 1
            print(f'Adding 1 to raise to power={power_base2}')
        # calculate actual number of zeros based on the valid power and data length N
    n_total = 2**power_base2
    print(bcolors.OKGREEN + "This corresponds to a total number of points of: " + str(n_total) + bcolors.ENDC)
    number_of_zeros_after = n_total - len(y_array)
    print(bcolors.OKGREEN + "This corresponds to a total number of zeros add after the last datapoint of: " + str(n_total) + bcolors.ENDC)
    # perform and return zero padding of y-data
    return np.pad(y_array, number_of_zeros_after)

