# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 22:25:42 2026

@author: mktaj
"""

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
DELTA_T = 100 # s
U = 10 # m s^-1
H = 100 # m
H0 = 0.1 # m
SIGMA = 50 * 10**3 # m
G = 9.81 # m s^-2


# %% Model simulations

def sw_run_for_time(simulation_time: float,
                    sw_model: ShallowWater):
    
    for i in range(time_steps):
        sw_model.leap_frog_step()
        if (i+1) % 100 == 0:
            print('{} steps done'.format(i+1))
        
    return sw_model

target_time = DELTA_T * 1000
runs = []
np.seterr(over='raise') # to break when model fails

for dt in np.arange(0, 20) * DELTA_T/2 + 120:
    
    time_steps = int(target_time / dt)
    print('Running at dt = {:.0f} for {} steps'.format(dt, time_steps))
    
    x = np.arange(-1, IDIM+1) * DELTA_X
    h = ndim_gaussian(x, x0=np.median(x), sigma=SIGMA, h_mean=H, h0=H0)

    h_dataset = xr.DataArray(np.astype(h, np.float64), 
                             dims=('x'),
                             coords={'x':x},
                             )
    
    sw_model = ShallowWater(h_dataset, dt, DELTA_X, (True,), G, H, U)
    
    try:
        run = sw_run_for_time(target_time, sw_model)
        runs.append(sw_model)
    except FloatingPointError as e:
        print('Model breaks at timestep dt = {}'.format(dt))
        runs.append(sw_model)
        break
    
# %% Animation

def xarray_to_seaborn(dataset: xr.Dataset,
                      index: dict=None,
                      ) -> pandas.DataFrame:
    data = dataset.copy(deep=True)
    data['h'] -= 100
    data['u'] *= 1
    data['v'] *= 1
    data = data
    if 'y' in data:
        data = data[{'y': slice(1,-1)}]
    if index is not None:
        data =  data[index]
    
    data = data.to_pandas().reset_index()
    data = data.melt(id_vars='x', var_name='Variable', value_name='h-100m, u, v')
    
    return data

    
def animate(model: ShallowWater):
    fig, ax = plt.subplots()
    ax.set_ylim(-0.02, 0.12)

    data_animation = xarray_to_seaborn(model.data, {'t': 1})

    plot = sns.lineplot(data=data_animation,
                        x=data_animation.columns[0],
                        y=data_animation.columns[2],
                        hue=data_animation.columns[1],
                        ax=ax)
    
    speed = 4
    frames = int(model.data.sizes['t'] / speed)
    fps = 30 * 100/model.DT
    
    def update(frame):
        data_animation = xarray_to_seaborn(model.data, {'t': frame*speed})
        ax.clear()
        plot = sns.lineplot(data=data_animation,
                            x=data_animation.columns[0],
                            y=data_animation.columns[2],
                            hue=data_animation.columns[1],
                            ax=ax)
        ax.set_ylim(-0.02, 0.12)

    ani = animation.FuncAnimation(fig, update, frames=frames, repeat=False)
    ani.save('animated_graph_{:.0f}.gif'.format(model.DT), 
             writer='pillow', fps=fps, dpi=300)

for run in runs:
    animate(run)
    
# %% Hovmöller diagram
    
def Hovmoller(sw_model: ShallowWater):
    data = sw_model.data['h'][{'x': slice(1,-1)}].copy(deep=True)
    t_max = sw_model.data.sizes['t']
    data = data.assign_coords(t=np.arange(0,t_max)*sw_model.DT / (60*60*24))
    plot = data.rename({'t': 't (days)'}).plot(yincrease=False, 
                                               aspect=0.5, size=10)
    plt.title('Timestep: {:.0f}s'.format(sw_model.DT))
    fig = plot.figure.savefig('Hovmoller_diagram_{:.0f}.png'.format(
            sw_model.DT), dpi=300)

for run in runs:
    Hovmoller(run)
    
sw_model = runs[-2]
data = sw_model.data['h'][{'x': slice(1,-1)}].copy(deep=True)
median = data['t'].median()
around_median = (int(median - 10), int(median + 10))
around_peak = data[{'x': 0, 't': slice(*around_median)}]
around_peak = around_peak.assign_coords(t=np.arange(*around_median)*sw_model.DT)
peak_index = around_peak.idxmax(dim='t')

print('Peak found at index: {:.0f} correspoding to time {:.0f}s'.format(
            peak_index/sw_model.DT, peak_index))
print('This corresponds to speed of {:.2f} m/s'.format(
            1.5*IDIM*DELTA_X/peak_index))

print('The Courant number is {:.2f}'.format(
        1.5*IDIM*DELTA_X/peak_index * sw_model.DT/DELTA_X))