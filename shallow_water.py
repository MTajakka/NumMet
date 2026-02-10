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

def meshgrid(dims: Union[Sequence[int], int],
              d_dims: Union[Sequence[float], float],
              ) -> Tuple[np.ndarray, ...]:
    """
    Parameters
    ----------
    dims : int or Sequence of ints
        Number of grid points in each axis. If a single value is given, it 
        will be used for all axis
    d_dims : float or Sequence of floats
        Spacing between points in each axis. If a single value is given, it
        will be used for all axis

    Raises
    ------
    ValueError
        Error raised if the dimensions are different between dims and d_dims.

    Returns
    -------
    X1, X2, ..., XN : tuple of ndarrays
        Same as numpy.meshgrid produces

    """
    
    if not isinstance(dims, Sequence) and not isinstance(d_dims, Sequence):
        dims = (dims,)
        d_dims = (d_dims,)
    elif not isinstance(dims, Sequence):
        dims = (dims,) * len(d_dims)
    elif not isinstance(d_dims, Sequence):
        d_dims = (d_dims,) * len(dims)
    elif len(dims) != len(d_dims):
        raise ValueError('Different amount of dimensions between dims, d_dims')
    
    axis = [np.arange(0, i_dim, 1) * dx for i_dim, dx in zip(dims, d_dims)]
    return np.meshgrid(*axis)

# =============================================================================
# def index_on_axis(i, axis: int, n_dim: int):
#     if axis > n_dim:
#         raise ValueError('axis out of range')
#     # s = (slice(None),)* n_dim
#     # s[axis] = i
#     return (slice(None),)*axis + (i,) + (slice(None),)*(n_dim-axis-1)
# 
# def shape_normal(shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
#     if axis > len(shape):
#         raise ValueError('axis out of range')
#     return shape[:axis] + shape[axis+1:]
# 
# def roll_diff(x: np.ndarray,
#               cyclic: bool=False,
#               axis: int=0,
#               boundary: Union[Tuple[np.ndarray, np.ndarray], float]=0.0
#               ) -> np.ndarray:
#     
#     x_left = np.roll(x, 1, axis=axis)
#     x_right = np.roll(x, -1, axis=axis)
#     
#     if not cyclic:
#         boundary_shape = shape_normal(x.shape, axis)
#         
#         if np.isscalar(boundary):
#             boundary = np.full((2,) + boundary_shape, boundary)
#         elif boundary[0].shape != boundary[1].shape:
#             raise ValueError('Boundary shapes do not match')
#         elif boundary[0].shape != boundary_shape:
#             raise ValueError('Given boundary doesnt match shape normal to axis')
#         
#         n_dim = len(x.shape)
#         x_left[index_on_axis(0, axis, n_dim)] = boundary[0]
#         x_right[index_on_axis(-1, axis, n_dim)] = boundary[1]
#         
#     return 1/2 * (x_right - x_left)
# =============================================================================

# =============================================================================
# def tmp_sum(n, **kwargs):
#     print(n)
#     print(kwargs)
#     return np.sum(n, **kwargs)
# 
# def roll_diff(k, **kwargs):
#     return 1/2 * (k[-1] - k[0])
# =============================================================================

# %% Shallow Water model

class ShallowWater:
    DT: float
    DX: float
    DY: float
    G: float
    # time: np.ndarray
    data: xr.Dataset
    # AXIS_X: int
    # AXIS_Y: int
    
    CYCLIC_X: bool
    CYCLIC_Y: bool
    # N_DIMS: int
    
    F: np.ndarray
    H_MEAN: float
    U_MEAN: float
    
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
        self.data['u'] = self._zeros()
        self.data['v'] = self._zeros()
        
        
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
        
        # self.euler_step()
    
    
    @property
    def now(self) -> xr.Dataset:
        return self.data.isel(t=-1)
    
    @property
    def old(self) -> xr.Dataset:
        return self.data.isel(t=-2)
    
    def _zeros(self) -> xr.DataArray:
        return xr.zeros_like(self.now['h'])
    
    @property
    def sizes(self) -> dict:
        return self.now.sizes
    
    
    def _handel_boundary(self, boundary, dim: str):
        if not isinstance(boundary, Sequence):
            boundary = (boundary,)*2
        if len(boundary) != 2:
            raise ValueError('Incorrect amount of boundaries')
        if not isinstance(boundary[0], xr.DataArray):
            left = xr.full_like(self.now['h'].isel({dim: 0}), boundary[0])
            left = left.drop_vars(dim)
            
            right = xr.full_like(self.now['h'].isel({dim: -1}), boundary[1])
            right = right.drop_vars(dim)
            
            boundary = (left, right)
        
        return boundary
        
    def _dn(self, 
            field: str, 
            dim: str,
            boundary: Union[Tuple[xr.DataArray, xr.DataArray],
                            Tuple[float, float],
                            xr.array,
                            float]=0.0,
            ) -> xr.DataArray:
        
        
        cyclic = self.CYCLIC_X if dim=='x' else self.CYCLIC_Y
        dn = self.DX if dim=='x' else self.DY
        
        if cyclic:
            boundary = (self.now[field].isel({dim: -1}).drop_vars(dim), 
                        self.now[field].isel({dim: 0}).drop_vars(dim))
        else:
            boundary = self._handel_boundary(boundary, dim=dim)
        expanded = xr.concat((boundary[0], 
                              self.now[field].drop_vars(dim), 
                              boundary[1]), 
                             dim=dim)
        
        left = expanded.isel({dim: slice(None,-2)})
        right = expanded.isel({dim: slice(2,None)})
        
        res = 1/(2 * dn) * (right - left)
        
        return res.assign_coords({dim: sw.now.coords[dim]})
    
    def _dy(self, field: np.ndarray) -> np.ndarray:
        # df_y = roll_diff(field, axis=self.AXIS_Y, cyclic=self.CYCLIC_Y)
        return #df_y / self.DY
    
    def _1D_slope(self) -> np.ndarray:
        return -self.F / self.G * self.U_MEAN
    
    def _tendencies(self) -> np.ndarray:
        '''
        I'm calculating all the tendecies in same function, because they 
        share mutiple derviatives between them
        '''    
        # aliases
        u = self.now['u']
        v = self.now['v']
        h = self.now['h']
        
        du_dx = self._dx(u)
        # du_dy = self._dy(u)
        du_dy = self._zeros()
        
        dv_dx = self._dx(v)
        # dv_dy = self._dy(v)
        dv_dy = self._zeros()
        
        dh_dx = self._dx(h)
        # dh_dy = self._dy(h)
        dh_dy = self._1D_slope() # 1D case
        
        tendecies = np.zeros_like(self.now)
        tendecies['u'] -= u*du_dx + v*du_dy - self.F*v + self.G*dh_dx
        tendecies['v'] -= u*dv_dx + v*dv_dy + self.F*u + self.G*dh_dy
        tendecies['h'] -= u*dh_dx + v*dh_dy + h*du_dx + h*dv_dy
        
        return tendecies

    
    def _add_new_step(self, fields_new: np.ndarray) -> None:
        self.time = np.vstack(self.time, fields_new)
        
    def euler_step(self) -> None:
        tendecies = self._tendencies()
        new_step = self.now + self.DT * tendecies
        self._add_new_step(new_step)
    
    def leap_frog_step(self) -> None:
        tendecies = self._tendencies()
        new_step = self.old + 2*self.DT * tendecies
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
                      cyclic=False, 
                      g=9.81, 
                      h_mean=10, 
                      u_mean=10)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    