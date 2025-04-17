import numpy as np
import pandas as pd

def mul_datasets(dataset, factor):
    assert type(factor) is float
    for df in dataset:
        if factor != 0.0:
            df.y *= factor
            print("You choose to multiply df(s) for: " + str(factor))
        else:
            raise ValueError
    return dataset

def normalise_dfs(dataset, xstart, xend):
    if xstart == "min" and xend == "max": 
        normalisation_coeff = 0
        for df in dataset:
            xstart = df.x.min()
            xend = df.x.max()
            normalisation_coeff = max(
            normalisation_coeff, df[df.x.between(xstart, xend)].y.max())
        for df in dataset:
                df.y = df.y / normalisation_coeff
        print("The normalisation coefficient is: " + str(normalisation_coeff))
    if xstart == "min" and type(xend) is float: 
        normalisation_coeff = 0
        for df in dataset:
            xstart = df.x.min()
            normalisation_coeff = max(
            normalisation_coeff, df[df.x.between(xstart, xend)].y.max())
        for df in dataset:
                df.y = df.y / normalisation_coeff
        print("The normalisation coefficient is: " + str(normalisation_coeff))
    if xend == "max" and type(xstart) is float:
        normalisation_coeff = 0
        for df in dataset:
            xend = df.x.max()
            normalisation_coeff = max(
            normalisation_coeff, df[df.x.between(xstart, xend)].y.max())
        for df in dataset:
                df.y = df.y / normalisation_coeff
        print("The normalisation coefficient is: " + str(normalisation_coeff))
    if type(xstart) is float and type(xend) is float:
        assert xstart < xend 
        normalisation_coeff = 0
        for df in dataset:
            normalisation_coeff = max(
            normalisation_coeff, df[df.x.between(xstart, xend)].y.max())
        for df in dataset:
            df.y = df.y / normalisation_coeff
        print("The normalisation coefficient is: " + str(normalisation_coeff))
    return dataset

def mul_dataset(df, factor):
    assert type(factor) is float
    if factor != 0.0:
        df.y *= factor
        print("You choose to multiply df for: " + str(factor))
    else:
        raise ValueError
    return df


def normalise_individual_dfs(dataset, xstart, xend):
    if xstart == "min" and xend == "max": 
        for df in dataset:
            xstart = df.x.min()
            xend = df.x.max()
            normalisation_coeff = df[df.x.between(xstart, xend)].y.max()
            print("The normalisation coefficient is: " + str(normalisation_coeff))
            df.y = df.y / normalisation_coeff
    if xstart == "min" and type(xend) is float: 
        for df in dataset:
            xstart = df.x.min()
            normalisation_coeff = df[df.x.between(xstart, xend)].y.max()
            print("The normalisation coefficient is: " + str(normalisation_coeff))
            df.y = df.y / normalisation_coeff
    if xend == "max" and type(xstart) is float: 
        for df in dataset:
            xend = df.x.max()
            normalisation_coeff = df[df.x.between(xstart, xend)].y.max()
            print("The normalisation coefficient is: " + str(normalisation_coeff))
            df.y = df.y / normalisation_coeff
    if type(xstart) is float and type(xend) is float:
        assert xstart < xend
        for df in dataset:
            normalisation_coeff = df[df.x.between(xstart, xend)].y.max()
            print("The normalisation coefficient is: " + str(normalisation_coeff))
            df.y = df.y / normalisation_coeff
    return dataset