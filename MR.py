import pandas as pd
from pandas import DataFrame
import numpy as np
import matplotlib as plt

def mr_set(list):
    new_list = []
    for df in list:
        df.x = df.x
        rho_zero = df.y.iloc[0]
        df.y = ((df.y - rho_zero) / rho_zero)*100
        new_list.append(df)
        print(rho_zero)
    return new_list

def mr(df, name, save=True):
    df.x = df.x
    rho_zero = df.y.iloc[0]
    df.y = ((df.y - rho_zero) / rho_zero)*100
    print(rho_zero)
    if save is True:
        df.to_csv(name + '.csv', sep='\t', encoding='utf-8')
    return df