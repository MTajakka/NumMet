# -*- coding: utf-8 -*-
"""
Created on Fri Feb  6 15:02:58 2026

@author: mktaj
"""

import numpy as np
import seaborn as sns
import pandas
from pathlib import Path
from matplotlib import animation
from matplotlib import pyplot as plt

import sys
sys.path.append('..') # added parent to sys path so the linadv file can be imported
from shallow_water import ShallowWater, meshgrid, ndim_gaussian

# %% Globals

IDIM = 99
DELTA_X = 10000 # m
DELTA_T = 100 # s
U = 10 # m s^-1
H = 100 # m
H0 = 0.1 # m
SIGMA = 50 * 10**3 # m
G = 9.81 # m s^-2

# %% Model simulation

X, = meshgrid((IDIM,), DELTA_X)
h = ndim_gaussian(X, x0=np.median(X), sigma=SIGMA, h_mean=H, h0=H0)

sw_model = ShallowWater(h, DELTA_T, DELTA_X, (True,), G, H)

# %% plotting

