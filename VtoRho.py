import pandas as pd
from pandas import DataFrame
import numpy as np

def voltage_to_resistivity_single_file(df, current, l, w, t):
    assert type(current) is float
    assert type(l) is float
    assert type(w) is float
    assert type(t) is float
    # rho_xx = R A / L
    area = w * t
    df.x = df.x
    df.y = (df.y * area) / (l*current)
    return df


def voltage_to_transverse_resistivity_single_file(df, current, t):
    assert type(current) is float
    assert type(t) is float
    # rho_yx = R*t
    df.x = df.x
    df.y = (df.y * t) / current
    return df



def voltage_to_resistivity(list, current, l, w, t):
    assert type(current) is float
    assert type(l) is float
    assert type(w) is float
    assert type(t) is float
    # rho_xx = R A / L
    new_list = []
    for df in list:
        area = w * t
        df.x = df.x
        df.y = (df.y * area) / (l*current)
        new_list.append(df)
    return new_list


def voltage_to_transverse_resistivity(list, current, t):
    assert type(current) is float
    assert type(t) is float
    # rho_yx = R*t
    new_list = []
    for df in list:
        df.x = df.x
        df.y = (df.y * t) / current
        new_list.append(df)
    return new_list

