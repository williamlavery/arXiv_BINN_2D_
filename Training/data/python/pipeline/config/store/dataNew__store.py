
from scipy.interpolate import RegularGridInterpolator
import numpy as np

#  ============= diffusion functions ========================

def diffusion_func1(u, theta):
    return np.full_like(u, theta[0])

def diffusion_func2(u, theta):
    return theta[0] + theta[1] * u

def diffusion_func3(u, theta):
    return theta[0] + theta[1]* u**2

def diffusion_func4(u, theta):
    return theta[0] + theta[1] * (1 - np.exp(-theta[2] * u))


#  ============= growth functions ========================

def growth_func1(u, theta):
    return np.full_like(u, theta[0])/2

def growth_func2(u, theta):
    return (theta[0] + theta[1] * u)/2

def growth_func3(u, theta):
    return (theta[0] + theta[1] * u**2)/2

def growth_func4(u, theta):
    return (theta[0] + theta[1] * (1 - np.exp(-theta[2] * u)))/2

# =========== initial conditions ===========


# Initial condition

def ic1(x, y, amplitude=1.0):
    """
    2D cosine-squared bump, like a Gaussian, centered at the geometric
    centre of the domain and decaying smoothly towards the boundaries.

    ic1(x, y) = amplitude * cos^2(π (x - xc) / Lx) * cos^2(π (y - yc) / Ly)

    where xc, yc are the geometric centres of the x, y domains.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    # Geometric centre of the domain
    xc = 0.5 * (x.min() + x.max())
    yc = 0.5 * (y.min() + y.max())

    Lx = x.max() - x.min()
    Ly = y.max() - y.min()

    X, Y = np.meshgrid(x, y, indexing='ij')

    denomx = Lx *1
    denomy = Ly *1


    cos_x = np.cos(np.pi * (X - xc) / denomx)
    cos_y = np.cos(np.pi * (Y - yc) / denomy)

    return amplitude * (cos_x**2) * (cos_y**2)



def scratch2(u, x, y):
    """
    Resample the first two axes (x,y) of u to new grid (x,y)
    using bilinear interpolation.

    Parameters:
        u : array of shape (nx_orig, ny_orig, ...)
        x : new x grid, length nx_new
        y : new y grid, length ny_new

    Returns:
        u_new : array of shape (nx_new, ny_new, ...)
    """

    nx_orig, ny_orig = u.shape[:2]

    # Original normalized grids (0..1)
    x_orig = np.linspace(0, 1, nx_orig)
    y_orig = np.linspace(0, 1, ny_orig)

    x_new = np.linspace(0, 1, len(x))
    y_new = np.linspace(0, 1, len(y))

    # Interpolator expects points as (x,y)
    interpolator = RegularGridInterpolator(
        (x_orig, y_orig),
        u,
        method='linear',
        bounds_error=False,
        fill_value=None
    )

    # Create target grid
    X_new, Y_new = np.meshgrid(x_new, y_new, indexing='ij')
    pts = np.stack([X_new, Y_new], axis=-1)

    return interpolator(pts)




def RDEq(x1, x2, t, u0, clear=False):
    return PDE_sim_old_2d(PDE_RHS_1D, u0, x1, x2, t, diffusion_func, growth_func, clear=clear)