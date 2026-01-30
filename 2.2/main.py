import numpy as np
import seaborn as sns
import pandas
from pathlib import Path

import sys
sys.path.append('..') # added parent to sys path so the linadv file can be imported
from linadv import (gaussian_distribution, 
                     height_tendency,
                     foward_euler_step,
                     leap_frog_step)

# %% Globals

IDIM = 99
DELTA_X = 10000 # m
X_MAX = (IDIM-1) * DELTA_X
DELTA_T = 300 # s
U = 10 # m s^-1

# %% Initial setup

x = np.linspace(0, X_MAX, IDIM)
h = gaussian_distribution(x, x0=X_MAX/2, h0=10, sigma=100000)
h_tend = height_tendency(h, u=U, dx=DELTA_X)

# %% Time evolution
TIME_STEPS = 100


h_time = np.array([h[:]]) # axis: 0: time, 1: x; t = 0
h_time = np.vstack((h_time, foward_euler_step(h, # t = 1
                                              dx=DELTA_X,
                                              dt=DELTA_T,
                                              u=U,)))

for i in range(TIME_STEPS - 1):
    h_now = h_time[-1]
    h_old = h_time[-2]
    h_next = leap_frog_step(h_now, h_old, 
                            dx=DELTA_X,
                            dt=DELTA_T,
                            u=U)
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
exercise = exercise.replace('.','_')

# %% Plotting

data = pandas.DataFrame({'x': x,
                         '0': h_time[0],
                         '10': h_time[10],
                         '100': h_time[100],
                         # '150': h_time[150],
                         })

data = data.melt(id_vars='x', var_name='Time', value_name='h')

sns.set_theme()
plot = sns.relplot(data=data,
            x='x',
            y='h',
            hue='Time',
            kind="line")
fig = plot.fig.savefig(exercise + '.pdf')

