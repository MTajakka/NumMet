# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 12:27:43 2026

@author: mktaj
"""

import numpy as np
import numpy.typing as npt
import seaborn as sns
import pandas
from pathlib import Path

# %% Globals

IDIM = 99
DELTA_X = 10000 # m
X_MAX = IDIM * DELTA_X
DELTA_T = 300 # s
U = 10 # m s^-1

# %%

def h(x: npt.ArrayLike, 
      x0: float = X_MAX/2,
      h0: float = 10, 
      sigma: float = 100000
      ) -> np.ndarray:
    
    return h0 * np.exp(-np.pow(x-x0, 2) / (2*sigma**2))


CENTRAL_DIFFERENCE = 1 / (2) * np.array([-1, 0, 1])

def h_tendency(h: npt.ArrayLike) -> np.ndarray:
    return -U / DELTA_X * np.convolve(h, CENTRAL_DIFFERENCE, mode='valid')


x = np.linspace(0, X_MAX, IDIM)
h_x = h(x)
h_tend = h_tendency(h_x)


# %% Excersise name from parent folder
''''
This is just so I dont have to always rename the plots 
to correct exercise
'''
path = Path().absolute() 
index = (path.parts.index('NumMet') 
         if 'NumMet' in path.parts 
         else path.parts.index('nummet'))
exercise = path.parts[index+1]
exercise = exercise.replace('.','_')

# %% Plotting

data = pandas.DataFrame({'x': x,
                         'h': h_x,
                         'h_tend': np.pad(h_tend, (1,1))})

sns.set_theme()
plot = sns.relplot(data=data,
            x=data.columns[0],
            y=data.columns[1],
            kind="line")
fig = plot.fig.savefig(exercise + '_1.pdf')

plot = sns.relplot(data=data,
            x=data.columns[0],
            y=data.columns[2],
            kind="line")
fig = plot.fig.savefig(exercise + '_2.pdf')