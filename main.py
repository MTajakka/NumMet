import numpy as np
import seaborn as sns
import xarray as xr
import pandas
from pathlib import Path
from matplotlib import animation
from matplotlib import pyplot as plt

import sys
sys.path.append('..') # added parent to sys path so the linadv file can be imported
from shallow_water import ShallowWater, ndim_gaussian

# %% Globals

IDIM = 99
DELTA_X = 10000 # m
DELTA_T = 50 # s
U = 10 # m s^-1
H = 100 # m
H0 = 0.1 # m
SIGMA = 50 * 10**3 # m
G = 9.81 # m s^-2


# %% Model simulations

x = np.arange(-1, IDIM+1) * DELTA_X
y = np.arange(-1, IDIM+1) * DELTA_X

h = ndim_gaussian(*np.meshgrid(x,y), 
                  x0=(np.median(x) - 100000, np.median(y)- 200000), 
                  sigma=SIGMA, 
                  h_mean=H, 
                  h0=H0)

h_dataset = xr.DataArray(h, 
                         dims=('y', 'x'),
                         coords={'x': x, 'y': y})

sw_model = ShallowWater(h_dataset, DELTA_T, DELTA_X, (True, True), G, H, U)

for i in range(5):
    for j in range(100):
        sw_model.leap_frog_step()
    print('{} steps done'.format((i+1)*100))
