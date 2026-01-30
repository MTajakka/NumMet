import numpy as np
import numpy.typing as npt


# %% 2.1

def gaussian_distribution(x: npt.ArrayLike, 
      x0: float = 0,
      h0: float = 10, 
      sigma: float = 10
      ) -> np.ndarray:
    return h0 * np.exp(-np.pow(x-x0, 2) / (2*sigma**2))

def convolve_diff(x: npt.ArrayLike) -> np.ndarray:
    CEN_DIFF_MASK = 1 / (2) * np.array([-1, 0, 1])
    return np.convolve(x, CEN_DIFF_MASK, mode='valid')

def roll_diff(x: npt.ArrayLike,
              cyclic: bool=False,
              axis: int=0) -> np.ndarray:
    x_left = np.roll(x, 1, axis=axis)
    x_right = np.roll(x, -1, axis=axis)
    if not cyclic:
        x_left[-1] = 0
        x_right[0] = 0
        
    return 1/2 * (x_right - x_left)
    

def central_diff(x: npt.ArrayLike,
                 **kwargs
                 ) -> np.ndarray:
    return roll_diff(x, **kwargs)

def height_tendency(x: npt.ArrayLike,
                    u: float=10,
                    dx: float=0.1,
                    **kwargs
                    ) -> np.ndarray:
    return -u/dx * central_diff(x, **kwargs)


# %% 2.2

def foward_euler_step(x_now: npt.ArrayLike,
                      dt: float=0.1,
                      **kwargs) -> np.ndarray:
    return x_now + dt * height_tendency(x_now, **kwargs)

def leap_frog_step(x_now: npt.ArrayLike,
                   x_old: npt.ArrayLike,
                   dt: float=0.1,
                   **kwargs) -> np.ndarray:
    h_tend = height_tendency(x_now, **kwargs)
    return x_old + 2 * dt * h_tend


