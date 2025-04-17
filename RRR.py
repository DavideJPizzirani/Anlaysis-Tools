import numpy as np
import pandas as pd
import math as ma
import matplotlib.pyplot as plt
import colorcet as cc

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


def rrr_calculator(dataset, min_T, max_T):
    assert type(min_T) is float
    assert type(max_T) is float
    if min_T > max_T:
        print(bcolors.FAIL + "Invalid input. Minimum temperature can not exceed maximum temperature!" + bcolors.FAIL)
    else:
        list_rrr = []
        list_index_sample = []
        i = 0
        for df in dataset:
            dist_to_x_min = (df.x - min_T).abs()
            closest_x_idx_min = dist_to_x_min.idxmin()
            y_min = df.y.loc[closest_x_idx_min]
            dist_to_x_max = (df.x - max_T).abs()
            closest_x_idx_max = dist_to_x_max.idxmin()
            y_max = df.y.loc[closest_x_idx_max]
            rrr = y_max/y_min
            x = i + 1
            print(str(rrr))
            list_rrr.append(rrr)
            list_index_sample.append(x)
        for rrr_values in list_rrr:
            plt.plot(x, rrr_values)
            plt.show()
    return list_rrr
        