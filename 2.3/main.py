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


CEN_DIFF_MASK = 1 / (2) * np.array([-1, 0, 1])
def central_diff(x: npt.ArrayLike) -> np.ndarray:
    return np.convolve(x, CEN_DIFF_MASK, mode='valid')

def h_tendency(h: npt.ArrayLike) -> np.ndarray:
    return -U / DELTA_X * central_diff(h)


x = np.linspace(0, X_MAX, IDIM)
h_x = h(x)
h_tend = h_tendency(h_x)
h_tend = np.pad(h_tend, (1,1))

# %% Time evolution
TIME_STEPS = 3000


h_time = np.array([h_x[:]]) # axis: 0: time, 1: x; t = 0
h_time = np.vstack((h_time, h_x[:] + DELTA_T * h_tend)) # t = 1

for i in range(TIME_STEPS - 1):
    h_now = h_time[-1]
    h_tend = h_tendency(np.concatenate(([h_now[-1]], h_now, [h_now[0]])))
    h_old = h_time[-2]
    h_next = h_old + 2 * DELTA_T * h_tend
    h_time = np.vstack((h_time, h_next))


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

# %% Plotting

data = {time: h_time[time] for time in range(0, 2880+320, 320)}
# data = {0: h_time[0],
#         320: h_time[320],
#         1600: h_time[1600]}
data['x'] = x
data = pandas.DataFrame(data)

data = data.melt(id_vars='x', var_name='Time', value_name='h')

sns.set_theme()
plot = sns.relplot(data=data,
            x='x',
            y='h',
            hue='Time',
            legend="full",
            kind="line")
fig = plot.fig.savefig(exercise + '.pdf')

