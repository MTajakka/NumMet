import numpy as np

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

def index_on_axis(i, axis: int, n_dim: int):
    if axis > n_dim:
        raise ValueError('axis out of range')
    # s = (slice(None),)* n_dim
    # s[axis] = i
    return (slice(None),)*axis + (i,) + (slice(None),)*(n_dim-axis-1)

def shape_normal(shape: Tuple[int, ...], axis: int) -> Tuple[int, ...]:
    if axis > len(shape):
        raise ValueError('axis out of range')
    return shape[:axis] + shape[axis+1:]

def roll_diff(x: np.ndarray,
              cyclic: bool=False,
              axis: int=0,
              boundary: Union[Tuple[np.ndarray, np.ndarray], float]=0.0
              ) -> np.ndarray:
    
    x_left = np.roll(x, 1, axis=axis)
    x_right = np.roll(x, -1, axis=axis)
    
    if not cyclic:
        boundary_shape = shape_normal(x.shape, axis)
        
        if np.isscalar(boundary):
            boundary = np.full((2,) + boundary_shape, boundary)
        elif boundary[0].shape != boundary[1].shape:
            raise ValueError('Boundary shapes do not match')
        elif boundary[0].shape != boundary_shape:
            raise ValueError('Given boundary doesnt match shape normal to axis')
        
        n_dim = len(x.shape)
        x_left[index_on_axis(0, axis, n_dim)] = boundary[0]
        x_right[index_on_axis(-1, axis, n_dim)] = boundary[1]
        
    return 1/2 * (x_right - x_left)

# %% Shallow Water model

class ShallowWater:
    DT: float
    DX: float
    DY: float
    G: float
    now: np.ndarray
    old: np.ndarray
    time: np.ndarray
    AXIS_X: int
    AXIS_Y: int
    
    CYCLIC_X: bool
    CYCLIC_Y: bool
    N_DIMS: int
    
    F: np.ndarray
    H_MEAN: float
    U_MEAN: float
    
    def __init__(self, 
                 h: npt.ArrayLike,
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
        
        self.now = np.zeros(h.shape, dtype=[('h','d'), ('u','d'), ('v','d')])
        self.now['h'] = h
        self.F = np.zeros_like(h)
        
        self.N_DIMS = len(np.shape(h))
        if self.N_DIMS > 2:
            raise NotImplementedError('There isnt a shallow water model for '
                                      + 'higer than two dimensions')
        
        if self.N_DIMS == 1:
            self.AXIS_X = 0,
            self.AXIS_Y = None
        else:
            self.AXIS_X = 1
            self.AXIS_Y = 0
        
        if not isinstance(d_dim, Sequence):
            d_dim = (d_dim) * self.N_DIMS
        if len(d_dim) != self.N_DIMS:
            raise ValueError('Dimensions of d_dim does not match models own')
        self.DX = d_dim[0]
        self.DY = d_dim[1]
        if not isinstance(cyclic, Sequence):
            cyclic = (cyclic) * self.N_DIMS
        if len(cyclic) != self.N_DIMS:
            raise ValueError('Dimensions of cyclic does not match models own')
        self.CYCLIC_X = cyclic[0]
        self.CYCLIC_Y = cyclic[1]
        
        self.initialize_time()
    
    def shape(self):
        return self.now.shape
    
    def _dx(self, field: np.ndarray):
        df_x = roll_diff(field, axis=self.AXIS_X, cyclic=True)
        return df_x / self.DX
    
    def _dy(self, field: np.ndarray):
        df_y = roll_diff(field, axis=self.AXIS_Y, cyclic=True)
        return df_y / self.DY
    
    def _1D_slope(self):
        return -self.F / self.G * self.U_MEAN
    
    def _zeros(self):
        return np.zeros(self.shape(), dtype='d')
    
    def _tendency(self):
        u_tend = self.zeros()
        v_tend = self.zeros()
        h_tend = self.zeros()
        
        du_dx = self._dx(self.now['u'])
        # du_dy = self._dy(self.now['u'])
        
        dv_dx = self._dx(self.now['v'])
        # dv_dy = self._dy(self.now['v'])
        
        dh_dx = self._dx(self.now['h'])
        # dh_dy = self._dy(self.now['h'])
        dh_dy = self._1D_slope()
        
# =============================================================================
#     def _u_tendency(self):
#         u_tend = self.zeros()
#         
#         du_x = roll_diff(self.now['u'], axis=self.AXIS_X, cyclic=True)
#         u_tend -= self.now['u'] * du_x / self.DX
#         
#         u_tend -= -self.F * self.now['v']
#         
#         dh_x = roll_diff(self.now['h'], axis=self.AXIS_X, cyclic=True)
#         u_tend -= self.G * dh_x / self.DX
#         
#         return u_tend
#         
#     def _v_tendency(self):
#         v_tend = self.zeros()
#         
#         dv_x = roll_diff(self.now['v'], axis=self.AXIS_X, cyclic=True)
#         v_tend -= self.now['u'] * dv_x / self.DX
#         
#         v_tend -= self.F * self.now['u']
#         
#         dh_y = self._1D_slope()
#         v_tend -= self.G * dh_y / self.DY
#         
#         return v_tend
#     
#     def _h_tendency(self):
#         h_tend = self.zeros()
#         
#         dh_x = roll_diff(self.now['h'], axis=self.AXIS_X, cyclic=True)
#         h_tend -= self.now['u'] * dh_x / self.DX
#         
#         dh_y = self._1D_slope()
#         h_tend -= self.now['v'] * dh_y / self.DY
# =============================================================================
        

        
        
        
        
        
    def euler_step(self):
        pass
    
    def leap_frog_step(self):
        pass
    
    def initialize_time(self):
        self.time = np.array([self.now[:]])
        self.euler_step()
