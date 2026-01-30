import numpy as np
import numpy.typing as npt
import seaborn as sns
import pandas
from pathlib import Path

import sys
sys.path.append('..')
from physics import gaussian_distribution, height_tendency

# %% Globals

IDIM = 99
DELTA_X = 10000 # m
X_MAX = (IDIM-1) * DELTA_X
DELTA_T = 300 # s
U = 10 # m s^-1

# %% Physics

x = np.linspace(0, X_MAX, IDIM)
h = gaussian_distribution(x, X_MAX/2, 10, 100000)
h_tend = height_tendency(h, u=U, dx=DELTA_X)

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
                         'h': h,
                         'h_tend': h_tend})

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