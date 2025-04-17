#%run nameofthescript (NO EXTENSION NEEDED)

#-------------------------#
    
#     GENERAL IMPORT
#           -
#       Libraries

#-------------------------#   

import os
import json
import pandas as pd
import numpy as np
import warnings
from numpy import float64
import scipy as py 
import math as ma
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import scipy.interpolate as scintpl
from numpy.polynomial.polynomial import polyfit
from numpy.polynomial import Polynomial
import ipywidgets as widgets
from ipywidgets import interact, fixed, widgets
from IPython.display import display
import matplotlib.colors as mcolors
from matplotlib import rcParams
from matplotlib.ticker import AutoLocator, AutoMinorLocator
import colorcet as cc
from typing import Any, Match, Optional, TextIO
from memspectrum import MESA
import scipy.signal as scs
import scipy.fftpack as scf
import scipy.fft as scfft  # type: ignore
import glob
from pathlib import Path
from lmfit import Model
from lmfit import conf_interval, conf_interval2d, report_ci
import numpy as np
import scipy
from scipy.optimize import curve_fit
from scipy.stats import chisquare
import scipy.optimize as opt
import statistics
import matplotlib.pylab as plt
import copy
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
from lmfit import Minimizer, create_params, fit_report

plt.style.use('dark_background')

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
