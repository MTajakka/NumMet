import numpy as np
import seaborn as sns
import pandas
from pathlib import Path
from matplotlib import animation
from matplotlib import pyplot as plt

from physics import (gaussian_distribution, 
                     height_tendency,
                     foward_euler_step,
                     leap_frog_step)

# %% Globals

IDIM = 99
DELTA_X = 10000 # m
X_MAX = IDIM * DELTA_X
DELTA_T = 300 # s
U = 10 # m s^-1

# %% Initial setup

x = np.linspace(0, X_MAX, IDIM)
h = gaussian_distribution(x, X_MAX/2, 10, 100000)
h_tend = height_tendency(h, u=U, dx=DELTA_X)

# %% Time evolution
TIME_STEPS = 3000


h_time = np.array([h[:]]) # axis: 0: time, 1: x; t = 0
h_time = np.vstack((h_time, foward_euler_step(h, # t = 1
                                              dx=DELTA_X,
                                              dt=DELTA_T,
                                              u=U,
                                              cyclic=True)))

for i in range(TIME_STEPS - 1):
    h_now = h_time[-1]
    h_old = h_time[-2]
    h_next = leap_frog_step(h_now, h_old, 
                            dx=DELTA_X,
                            dt=DELTA_T,
                            u=U,
                            cyclic=True)
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

data_times = {time: h_time[time] for time in range(0, 2880+320, 320)}
# data = {0: h_time[0],
#         320: h_time[320],
#         1600: h_time[1600]}
data_times['x'] = x
data_times = pandas.DataFrame(data_times)

data_times = data_times.melt(id_vars='x', var_name='Time', value_name='h')

sns.set_theme()
plot = sns.relplot(data=data_times,
            x='x',
            y='h',
            hue='Time',
            legend="full",
            kind="line")
fig = plot.fig.savefig(exercise + '.pdf')

# %% Animation

fig, ax = plt.subplots()

data_animation = pandas.DataFrame({'x': x,
                                   'h': h_time[0]})
plot = sns.lineplot(data=data_animation,
            x='x',
            y='h',
            # kind="line",
            ax=ax)
def update(frame):
    data_animation = pandas.DataFrame({'x': x,
                                       'h': h_time[frame]})
    ax.clear()
    plot = sns.lineplot(data=data_animation,
                x='x',
                y='h',
                # kind="line",
                ax=ax)
    
ani = animation.FuncAnimation(fig, update, frames=300, repeat=False)
plt.show()