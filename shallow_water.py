import numpy as np
import xarray as xr

from typing import Tuple, Union, Sequence
import numpy.typing as npt

# %% General physics equations

def ndim_gaussian(*x: np.ndarray,
                  x0: Union[Sequence[float], float],
                  sigma: Union[Sequence[float], float],
                  h_mean: float,
                  h0: float,
                  ) -> np.ndarray:
    dims = len(x)
    
    if not isinstance(x0, Sequence):
        x0 = (x0,) * dims
    if not isinstance(sigma, Sequence):
        sigma = (sigma,) * dims
    
    if len(x0) != dims:
        raise ValueError('x0 has different dimesion than input')
    if len(sigma) != dims:
        raise ValueError('sigma has different dimesion than input')
        
    x = np.moveaxis(x, 0, -1) # x,y,.. have to be last for numpy broadcast 
    exp = np.pow(np.array(x) - x0, 2) / (2*np.pow(sigma, 2))
    return h_mean + h0 * np.exp(-np.sum(exp, axis=-1))

# %% Shallow Water model

class ShallowWater:
    DT: float
    DX: float
    DY: float
    G: float
    # time: np.ndarray
    data: xr.Dataset
    cache: dict
    # AXIS_X: int
    # AXIS_Y: int
    
    CYCLIC_X: bool
    CYCLIC_Y: bool
    # N_DIMS: int
    
    F: xr.DataArray
    H_MEAN: float
    U_MEAN: float
    
    now: xr.Dataset
    old: xr.Dataset
    
    def __init__(self, 
                 h: xr.DataArray,
                 dt: float,
                 d_dim: Union[Sequence[float], float],
                 cyclic: Union[Sequence[bool], bool],
                 g: float,
                 h_mean: float,
                 u_mean: float=10,
                 ):
        
        self.DT = dt
        self.H_MEAN = h_mean
        self.U_MEAN = u_mean
        self.G = g
        
        
        self.data = xr.Dataset({'h': h})
        self.data = self.data.expand_dims(dim='t')
        self.now = self.data[{'t': -1}] # cacheing latest step
        self.data['u'] = self._zeros().expand_dims(dim='t')
        self.data['v'] = self._zeros().expand_dims(dim='t')
        self.now = self.data[{'t': -1}] # adding u and v into now
        
        
        self.F = self._zeros()
        
        self.cache = dict()
        self._clear_cache()
        
        
        if not isinstance(d_dim, Sequence):
            d_dim = (d_dim,) * len(self.sizes)
            
        if len(d_dim) != len(self.sizes):
            raise ValueError('Dimensions of d_dim does not match models own')
            
        self.DX = d_dim[0]
        self.DY = d_dim[1] if len(d_dim) == 2 else None
        
        if not isinstance(cyclic, Sequence):
            cyclic = (cyclic,) * len(self.sizes)
            
        if len(cyclic) != len(self.sizes):
            raise ValueError('Dimensions of cyclic does not match models own')
            
        self.CYCLIC_X = cyclic[0]
        self.CYCLIC_Y = cyclic[1] if len(cyclic) == 2 else False
        
        self.euler_step()
    
    
# =============================================================================
#     @property
#     def now(self) -> xr.Dataset:
#         return self.data[{'t': -1}]
#     
#     @property
#     def old(self) -> xr.Dataset:
#         return self.data[{'t': -2}]
# =============================================================================
    
    def _zeros(self) -> xr.DataArray:
        return xr.zeros_like(self.now['h'])
    
    @property
    def sizes(self) -> dict:
        return self.now.sizes
    
    @property
    def dims(self):
        return self.now.dims
    
    def _clear_cache(self) -> None:
        self.cache = dict()
    
    def _generate_boundary(self, 
                           boundary: Union[Tuple[xr.DataArray, xr.DataArray],
                                           Tuple[float, float],
                                           xr.DataArray,
                                           float], 
                           dim: str):
        if not isinstance(boundary, Sequence):
            boundary = (boundary,)*2
        if len(boundary) != 2:
            raise ValueError('Incorrect amount of boundaries')
        if not isinstance(boundary[0], xr.DataArray):
            left = xr.full_like(self.now['h'][{dim: 0}], boundary[0])
            
            right = xr.full_like(self.now['h'][{dim: -1}], boundary[1])
            
            boundary = (left, right)
        
        return boundary
        
    def _central_diff(self, 
                      field: str, 
                      dim: str,
                      boundary: Union[Tuple[xr.DataArray, xr.DataArray],
                                      Tuple[float, float],
                                      xr.DataArray,
                                      float]=0.0,
                      cached: bool=True,
                      ) -> np.ndarray:
        
        if cached and field + dim in self.cache:
            return self.cache[field + dim]
        
        cyclic = self.CYCLIC_X if dim=='x' else self.CYCLIC_Y
        dn = self.DX if dim=='x' else self.DY
        
        if cyclic:
            boundary = (self.now[field][{dim: -2}], 
                        self.now[field][{dim: 1}])
        else:
            boundary = self._generate_boundary(boundary, dim=dim)
            
        field_data = self.now[field].copy(deep=True)
        field_data[{dim: 0}] = boundary[0]
        field_data[{dim: -1}] = boundary[1]

        
        # left = field_data[{dim: slice(None, -2)}].to_numpy()
        # right = field_data[{dim: slice(2, None)}].to_numpy()
        
        left = field_data.shift({dim: 1}).to_numpy()
        right = field_data.shift({dim: -1}).to_numpy()
        
        res = 1/(2 * dn) * (right - left)
        
        
        
        # res = res.assign_coords({dim: self.now.coords[dim][1:-1]})
        self.cache[field + dim] = res
        return res
        
    def _1D_slope(self) -> xr.DataArrayy:
        return -self.F / self.G * self.U_MEAN
    
    def u_tendency(self) -> xr.DataArray:
        # aliases
        
        u = self.now['u'].to_numpy()
        v = self.now['v'].to_numpy()
        
        du_dx = self._central_diff('u', 'x')
        # du_dy = self._central_diff('u', 'y')
        du_dy = self._zeros().to_numpy()
        dh_dx = self._central_diff('h', 'x')
        
        f = self.F.to_numpy()
        g = self.G
        
        return -(u*du_dx + v*du_dy - f*v + g*dh_dx)

    def v_tendency(self) -> xr.DataArray:
        # aliases
        u = self.now['u'].to_numpy()
        v = self.now['v'].to_numpy()
        
        dv_dx = self._central_diff('v', 'x')
        # dv_dy = self._central_diff('v', 'y')
        dv_dy = self._zeros().to_numpy()
        # dh_dy = self._central_diff('h', 'y')
        dh_dy = self._1D_slope().to_numpy()
        
        f = self.F.to_numpy()
        g = self.G
        
        return -(u*dv_dx + v*dv_dy + f*u + g*dh_dy)

    def h_tendency(self) -> xr.DataArray:
        # aliases
        u = self.now['u'].to_numpy()
        v = self.now['v'].to_numpy()
        h = self.now['h'].to_numpy()
        
        du_dx = self._central_diff('u', 'x')
        # dv_dy = self._central_diff('v', 'y')
        dv_dy = self._zeros().to_numpy()
        dh_dx = self._central_diff('h', 'x')
        # dh_dy = self._central_diff('h', 'y')
        dh_dy = self._1D_slope().to_numpy()
        
        return -(u*dh_dx + v*dh_dy + h*du_dx + h*dv_dy)
    
    def _update_now(self):
        self.now = self.data[{'t': -1}]
        self.old = self.data[{'t': -2}]
    
    def _add_new_step(self, new_dataset: xr.Dataset) -> None:
        self.data = xr.concat((self.data, new_dataset), dim='t')
        self._update_now()
        self._clear_cache()
        
    def euler_step(self) -> None:
        new_step = xr.zeros_like(self.now)
        
        new_u = self.now['u'].to_numpy() + self.DT * self.u_tendency()
        new_step['u'] = (self.dims, new_u)
        
        new_v = self.now['v'].to_numpy() + self.DT * self.v_tendency()
        new_step['v'] = (self.dims, new_v)
        
        new_h = self.now['h'].to_numpy() + self.DT * self.h_tendency()
        new_step['h'] = (self.dims, new_h)
        
        # tendecies = self._tendencies()
        # new_step = self.now + self.DT * tendecies
        self._add_new_step(new_step)
    
    def leap_frog_step(self) -> None:
        new_step = xr.zeros_like(self.now)
        
        new_u = self.old['u'].to_numpy() + 2*self.DT * self.u_tendency()
        new_step['u'] = (self.dims, new_u)
        
        new_v = self.old['v'].to_numpy() + 2*self.DT * self.v_tendency()
        new_step['v'] = (self.dims, new_v)
        
        new_h = self.old['h'].to_numpy() + 2*self.DT * self.h_tendency()
        new_step['h'] = (self.dims, new_h)
        
        # tendecies = self._tendencies()
        # new_step = self.old + 2*self.DT * tendecies
        self._add_new_step(new_step)
    

# %% File itself is run

if __name__ == '__main__':
    x = np.arange(10)
    y = np.arange(4)
    
    h = xr.DataArray(np.arange(40).reshape((10,4)), 
                     dims=('x', 'y'),
                     coords={'x':x, 'y':y},
                     )
    
    sw = ShallowWater(h, 
                      dt=10, 
                      d_dim=10, 
                      cyclic=True, 
                      g=9.81, 
                      h_mean=10, 
                      u_mean=10)