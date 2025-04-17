import glob
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
import warnings

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

def create_list_of_dfs(subfolder, xcolumn, ycolumn):
    for path, subdir, files in os.walk(subfolder):
        print(files)
        print("Number of files is: " + str(len(files)))
        list_dfs = []
    for file in files:
        list_dfs.append((pd.read_csv(file, sep="\t", header=0, usecols=[xcolumn, ycolumn], names=["x", "y"])))
    print(list_dfs)
    print(bcolors.OKGREEN +"Number of elements in the list is " + str(len(list_dfs)) + bcolors.OKGREEN)
    return list_dfs




