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

def s(x):
    return 4 * np.sin(x) / np.pi


x = np.linspace(0, 2 * np.pi, N)
data = pandas.DataFrame({'x': x,
                         's(x)': s(x)},
                        columns=['x', 's(x)'])

# %% Ploting

sns.set_theme()
plot = sns.relplot(data=data,
            x='x',
            y='s(x)')
fig = plot.fig.savefig('1.pdf')