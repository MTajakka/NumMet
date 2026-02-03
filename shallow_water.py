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

def roll_diff(x: npt.ArrayLike,
              cyclic: bool=False,
              axis: int=0) -> np.ndarray:
    x_left = np.roll(x, 1, axis=axis)
    x_right = np.roll(x, -1, axis=axis)
    if not cyclic:
        x_left[-1] = 0
        x_right[0] = 0
        
    return 1/2 * (x_right - x_left)

# %% Shallow Water model


class ShallowWater:
    DT: float
    D_DIM: Sequence[float]
    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    CYCLIC: Sequence[bool]
    N_DIMS: int
    
    F: np.ndarray
    H_MEAN: float
    U_MEAN: float
    
    t_h: np.ndarray
    t_u: np.ndarray
    t_v: np.ndarray
    
    
    def __init__(self, 
                 dt: float,
                 h: npt.ArrayLike,
                 d_dim: Union[Sequence[float], float],
                 cyclic: Union[Sequence[bool], bool],
                 h_mean: float,
                 u_mean: float,
                 ):
        self.DT = dt
        self.h = h
        self.u = np.zeros_like(h)
        self.v = np.zeros_like(h)
        self.F = np.zeros_like(h)
        self.N_DIMS = len(np.shape(h))
        if self.N_DIMS > 2:
            raise NotImplementedError('There isnt a shallow water model for '
                                      + 'higer than two dimensions')
        
        if not isinstance(d_dim, Sequence):
            d_dim = (d_dim) * self.N_DIMS
        self.D_DIM = d_dim
        if len(self.D_DIM) != self.N_DIMS:
            raise ValueError('Dimensions of d_dim does not match models own')
        if not isinstance(cyclic, Sequence):
            cyclic = (cyclic) * self.N_DIMS
        self.CYCLIC = cyclic
        if len(self.CYCLIC) != self.N_DIMS:
            raise ValueError('Dimensions of cyclic does not match models own')
        
        self.initialize_time()
        
    def euler_step(self):
        pass
    
    def leap_frog_step(self):
        pass
    
    def initialize_time(self):
        self.t_h = np.array([self.h[:]])
        self.t_u = np.array([self.u[:]])
        self.t_v = np.array([self.v[:]])
        
        self.euler_step()
