import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
sys.path.append('../../../') 

    
    
class OriginalData:

    def __init__(self, 
                 data,
                 plot=False):
       
        """
        Initialize the DataGenerator class.

        Parameters:
        - path_to_data (str): Path to the data directory.

        """


        self.plot = plot
        self.data = data 

        self.inputs =self.data["inputs"]
        self.u =self.data["outputs"]
        self.u_clean =self.data["clean"]
        self.u0 = self.u[..., 0]


        self.x = np.unique(self.inputs[:, 0]).copy()
        self.x_num = len(self.x)
        self.t = np.unique(self.inputs[:, 1]).copy()
        self.t_num = len(self.t)

        self.xmin = np.min(self.x)
        self.xmax = np.max(self.x)
        self.L = self.xmax - self.xmin
        self.nx = len(self.x)

        self.D = self.data["D"]
        self.r = self.data["r"]
        self.gamma =  0.2
        self.K = self.data['K']

        self.class_info = {"original?":1,
                          "DValue":self.D,
                          "rValue":self.r,
                          "gamma":self.gamma,
                          "K":self.K,
                          "xNum":self.x_num,
                          "tNum":self.t_num,
                          }

        if self.plot:
            self.plot_1d_data(self.x, self.t, self.u, self.u_clean)


    def plot_1d_data(x,t,u, u_clean):
            """
            Plot the initial condition heatmap.

            Parameters:
            - x1, x2 (np.ndarray): Spatial coordinate arrays.
            - u0 (np.ndarray): Initial condition data.
            """
            plt.figure(figsize=(8, 4))
            for tidx in range(len(t)):
                plt.plot(x, u[:, tidx], label=f'Noisy')
                plt.xlabel('x [mm]')
                plt.ylabel('u [cells/mm^2]')
                plt.title(f'Cell Density at t={t[tidx]:.2f} s')
            for tidx in range(len(t)):
                plt.plot(x, u_clean[:, tidx], label=f'Clean       t={t[tidx]:.2f} s', linestyle='--')
            plt.legend(ncols=2, fontsize=8)
            plt.tight_layout()
            plt.show()  


def add_noise(func_info, u_clean, gamma, noise_lvl = 0.01, seed = 0, t_start_idx=1):
    """
    Generate additive noise for the cell density.
    
    Parameters:
    - u_clean (np.ndarray): Noise-free cell density data.
    - gamma (float). Exponent from assumed noise model.
    - noise_lvl (float): Noise level to scale additive noise.
    - seed (float). Sets seed for normal distribution noise is drawn from.

    Returns:
    - np.ndarray: Noisy density data.
    """     
    u = u_clean.copy()
    shape = u_clean[...,t_start_idx:].shape  # drop IC
    np.random.seed(seed) 
    # clip to ensure noise always added

    noise = (u_clean[...,t_start_idx:]**gamma)
    u[..., t_start_idx:] += noise_lvl*noise*np.random.normal(size=shape)
    return u.clip(0,np.inf) # non-negative

#===================================================================================================
#===================================================================================================   
#===================================================================================================            


#===================================================================================================
# 2D DATA CLASS
#===================================================================================================

class Data:

    def __init__(self, 
                 x1,
                 x2,
                 t,
                 u_clean,
                 u,
                 theta_D,
                 theta_G,
                 K = 1,
                 gamma = 0.2,
                 plot=False):
        """
        2D analogue of Data class.

        Parameters:
        - x1, x2 (np.ndarray): 1D spatial grids.
        - t (np.ndarray): 1D time grid.
        - u_clean (np.ndarray): Noise-free data, shape (nx, ny, nt).
        - u (np.ndarray): Noisy data, shape (nx, ny, nt).
        - theta_D, theta_G: parameter vectors for your model.
        - K, gamma: scalars as in 1D version.
        - plot (bool): If True, call plot_2d_data on init.
        """

        self.plot = plot
        self.u = u
        self.u_clean = u_clean
        self.u0 = self.u[..., 0]  # initial condition u(x1,x2,t=0)

        self.x1 = x1
        self.x2 = x2
        self.t = t

        # Now inputs are (x1, x2, t) instead of (x1, t)
        self.inputs = generate_inputs_2d(self.x1, self.x2, self.t)

        self.xmin = np.min(self.x1)
        self.xmax = np.max(self.x1)
        self.ymin = np.min(self.x2)
        self.ymax = np.max(self.x2)

        self.Lx = self.xmax - self.xmin
        self.Ly = self.ymax - self.ymin

        self.nx = len(self.x1)
        self.ny = len(self.x2)
        self.nt = len(self.t)

        self.theta_D = theta_D
        self.theta_G = theta_G
        self.gamma = gamma
        self.K = K

        self.ClassInfo = {
            "xNum": len(x1),
            "yNum": len(x2),
            "tNum": len(t),
            "gamma": gamma,
            "K": K
        }

        if self.plot:
            plot_2d_data(self.x1, self.x2, self.t, self.u, self.u_clean)


#===================================================================================================
# 2D PLOTTING (ANALOGOUS TO 1D, BUT HEATMAPS)
#===================================================================================================

def plot_2d_data(x1, x2, t, u, u_clean, times_to_plot=None):
    """
    Plot 2D heatmaps of u(x1,x2,t) at a few time points.

    Parameters:
    - x1, x2 (np.ndarray): 1D spatial grids.
    - t (np.ndarray): 1D time grid.
    - u (np.ndarray): noisy data, shape (nx, ny, nt).
    - u_clean (np.ndarray): clean data, shape (nx, ny, nt).
    - times_to_plot (list of int, optional): indices into t to plot.
    """

    nx, ny, nt = u.shape
    assert nx == len(x1) and ny == len(x2) and nt == len(t), "Shape mismatch."

    if times_to_plot is None:
        # pick up to 3 representative time indices: start, middle, end
        times_to_plot = [0, nt // 2, nt - 1] if nt >= 3 else list(range(nt))

    n_plots = len(times_to_plot)

    fig, axes = plt.subplots(2, n_plots, figsize=(4 * n_plots, 8), squeeze=False)

    X, Y = np.meshgrid(x1, x2, indexing='ij')  # to label axes properly if needed

    for col, tidx in enumerate(times_to_plot):
        tt = t[tidx]

        # Noisy
        ax = axes[0, col]
        im = ax.imshow(u[:, :, tidx].T, origin='lower',
                       extent=[x1[0], x1[-1], x2[0], x2[-1]],
                       aspect='auto')
        ax.set_title(f"Noisy u(x1,x2,t) at t={tt:.2f}")
        ax.set_xlabel("x1 [mm]")
        ax.set_ylabel("x2 [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Clean
        ax = axes[1, col]
        im = ax.imshow(u_clean[:, :, tidx].T, origin='lower',
                       extent=[x1[0], x1[-1], x2[0], x2[-1]],
                       aspect='auto')
        ax.set_title(f"Clean u(x1,x2,t) at t={tt:.2f}")
        ax.set_xlabel("x1 [mm]")
        ax.set_ylabel("x2 [mm]")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


#===================================================================================================
# NOISE (WORKS FOR 1D OR 2D AS WRITTEN)
#===================================================================================================

def add_noise(u_clean, gamma, noise_lvl=15, seed=0, t_start_idx=1):
    """
    Generate additive noise for the cell density.

    Works for 1D, 2D, ... as long as time is the last axis.

    Parameters:
    - u_clean (np.ndarray): Noise-free cell density data (..., nt).
    - gamma (float): Exponent from assumed noise model.
    - noise_lvl (float): Noise level to scale additive noise.
    - seed (int): Seed for normal distribution noise is drawn from.

    Returns:
    - np.ndarray: Noisy density data, same shape as u_clean.
    """
    u = u_clean.copy()
    shape = u_clean[..., t_start_idx:].shape  # all spatial dims, times >= t_start_idx
    np.random.seed(seed)

    # clip to ensure noise always added
    u[..., t_start_idx:] += noise_lvl * (u_clean[..., t_start_idx:].clip(1, np.inf) ** gamma) \
                            * np.random.normal(size=shape)
    return u.clip(0, np.inf)  # non-negative


#===================================================================================================
# INPUT GENERATION (2D)
#===================================================================================================

def generate_inputs_2d(x1, x2, t):
    """ 
    Generates 3D model inputs (x1, x2, t) for 2D space + time.

    Parameters:
    - x1, x2, t (np.ndarray): 1D spatial and temporal grids.

    Returns:
    - np.ndarray: shape (nx * ny * nt, 3) with columns [x1, x2, t].
    """
    X, Y, T = np.meshgrid(x1, x2, t, indexing='ij')  # shapes (nx, ny, nt)

    return np.stack(
        [
            X.ravel(),   # x1-coordinates
            Y.ravel(),   # x2-coordinates
            T.ravel()    # time
        ],
        axis=-1
    )

