
import numpy as np
import pandas as pd

def eval_at_x(df, x):
    assert type(x) is float
    dist_to_x = (df.x - x).abs()
    closest_x_idx = dist_to_x.idxmin()
    y = df.y.loc[closest_x_idx]
    return y

def dataset_eval_at_x(dataset, x):
    assert type(x) is float
    y_per_df = []
    for df in dataset:
        dist_to_x = (df.x - x).abs()
        closest_x_idx = dist_to_x.idxmin()
        y = df.y.loc[closest_x_idx]
        y_per_df.append(y)
    return y_per_df