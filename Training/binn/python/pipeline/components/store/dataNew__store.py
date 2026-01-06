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
def ic1(x):
    mean_x = np.mean(x)
    L = np.max(x) - np.min(x)
    return 1/2 - 1/2 * np.cos(2 * np.pi * (x - mean_x) / L)


def scratch(u, x):
    """
    Resample the last axis of the input array `u` to have `num_x` points
    using linear interpolation.

    Parameters:
        u: np.ndarray
            Original data with shape (..., 38) assumed on the last axis.
        num_x: int
            Desired number of x-points in the output.

    Returns:
        np.ndarray with shape (..., num_x)
    """
    num_x = len(x)
    original_num_x = u.shape[-1]
    x_original = np.linspace(0, 1, original_num_x)
    x_new = np.linspace(0, 1, num_x)

    # Interpolate along the last axis
    interpolator = interp1d(x_original, u, axis=-1, kind='linear')
    return interpolator(x_new)



def RDEq(x, t, u0, clear=False):
    return PDE_sim_old(PDE_RHS_1D, u0, x, t, diffusion_func, growth_func, clear=clear)