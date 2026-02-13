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

# %% Model simulation

x = np.arange(-1, IDIM+1) * DELTA_X
h = ndim_gaussian(x, x0=np.median(x), sigma=SIGMA, h_mean=H, h0=H0)

h_dataset = xr.DataArray(h, 
                         dims=('x'),
                         coords={'x':x},
                         )

sw_model = ShallowWater(h_dataset, DELTA_T, DELTA_X, (True,), G, H, U)

for i in range(100):
    for j in range(100):
        sw_model.leap_frog_step()
    print('{} steps done'.format((i+1)*100))

# %% plotting

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
    

data = xarray_to_seaborn(sw_model.data, {'t': 0})


sns.set_theme()
plot = sns.relplot(data=data,
                   x=data.columns[0],
                   y=data.columns[2],
                   hue=data.columns[1],
                   kind="line")
plot.set(title='Time: {:.0f}s'.format(DELTA_T * 0))
fig = plot.fig.savefig('b.pdf')

data = xarray_to_seaborn(sw_model.data, {'t': -1})


plot = sns.relplot(data=data,
                   x=data.columns[0],
                   y=data.columns[2],
                   hue=data.columns[1],
                   kind="line")
plot.set(title='Time: {:.0f}s'.format(DELTA_T * (sw_model.data.sizes['t'] - 1)))
fig = plot.fig.savefig('c.pdf')

data = sw_model.data.copy(deep=True).drop_vars(('h', 'v', 'u'))
data['u_dh_dx'] = sw_model.now['u'] * sw_model._central_diff('h', 'x')
data['v_dH_dy'] = sw_model.now['v'] * sw_model._1D_slope()
data['h_du_dx'] = sw_model.now['h'] * sw_model._central_diff('u', 'x')

data = data[{'x': slice(1, -1)}]
data = data.to_pandas().reset_index()
data = data.melt(id_vars='x', var_name='Components', value_name='h_tend')

plot = sns.relplot(data=data,
                   x=data.columns[0],
                   y=data.columns[2],
                   hue=data.columns[1],
                   kind="line")
fig = plot.fig.savefig('d.pdf')

# %% 3.2 Hovmöller diagram

data = sw_model.data['h'][{'x': slice(1,-1), 
                           't': slice(None, 4000)}].copy(deep=True)
data = data.assign_coords(t=np.arange(0,4000)*DELTA_T / (60*60*24))
plot = data.rename({'t': 't (days)'}).plot(yincrease=False, 
                                           aspect=0.5, size=10)
fig = plot.figure.savefig('3_2.png', dpi=300)

around_peak = data[{'x': 0, 't': slice(400, 600)}]
around_peak = around_peak.assign_coords(t=np.arange(400,600)*DELTA_T)
peak_index = around_peak.idxmax(dim='t')

print('Peak found at index: {:.0f} correspoding to time {:.0f}s'.format(
            peak_index/DELTA_T, peak_index))
print('This corresponds to speed of {:.2f} m/s'.format(
            1.5*IDIM*DELTA_X/peak_index))

print('The Courant number is {:.2f}'.format(
        1.5*IDIM*DELTA_X/peak_index * DELTA_T/DELTA_X))

# %% Animation

fig, ax = plt.subplots()
ax.set_ylim(-0.02, 0.12)

data_animation = xarray_to_seaborn(sw_model.data, {'t': 1})

plot = sns.lineplot(data=data_animation,
                    x=data_animation.columns[0],
                    y=data_animation.columns[2],
                    hue=data_animation.columns[1],
                    ax=ax)

def update(frame):
    data_animation = xarray_to_seaborn(sw_model.data, {'t': frame*3})
    ax.clear()
    plot = sns.lineplot(data=data_animation,
                        x=data_animation.columns[0],
                        y=data_animation.columns[2],
                        hue=data_animation.columns[1],
                        ax=ax)
    ax.set_ylim(-0.02, 0.12)
    
ani = animation.FuncAnimation(fig, update, frames=300, repeat=False)
ani.save('animated_graph.gif', writer='pillow', fps=30, dpi=300)

