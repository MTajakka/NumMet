# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 11:31:55 2026

@author: mktaj
"""

import numpy as np
import seaborn as sns
import pandas

# %% Globals

N = 100 # number of points

# %% Calculations

def s(x, n: int):
    return 4 * np.sin(n * x) / (n * np.pi)

def sum_s(x, N: int):
    s_sum = np.zeros_like(x) # initial values similar to x
    for i in range(N):
        s_sum += s(x, i+1)
    return s_sum

x = np.linspace(0, 2 * np.pi, N)
data = pandas.DataFrame({'x': x,
                         'N = 1': sum_s(x, 1), 
                         'N = 7': sum_s(x, 7), 
                         'N = 151': sum_s(x, 151)})

data = data.melt(id_vars='x', var_name='N', value_name='s(x)')
# %% Ploting

sns.set_theme()
plot = sns.relplot(data=data,
            x='x',
            y='s(x)',
            hue='N')
fig = plot.fig.savefig('2.pdf')