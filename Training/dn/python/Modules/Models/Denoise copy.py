import sys, torch, pdb
sys.path.append('../../../../') # if inside dn/python/pipeline/exc
sys.path.append('../../../')
sys.path.append('../../')
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from dn.python.Modules.Models.BuildMLP2 import BuildMLP2
#from Modules.Utils.Gradient import Gradient


def data_loss_MSE(self, pred, true):
    return (pred - true) ** 2

def data_loss_GLS(self, pred, true):
    residual = (pred - true) ** 2
    residual *= pred.abs().clamp(min=1e-10) ** (-2*self.gamma)
    return residual


class u_MLP_dn(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for predicting the solution of a governing PDE.

    This model includes three hidden layers with 128 sigmoid-activated neurons.
    The output layer uses a softplus activation to ensure non-negative cell densities.

    Attributes:
        scale (float): Output scaling factor, defaults to carrying capacity.
        mlp (nn.Module): The MLP model constructed with the specified layers and activations.

    Args:
        input_features (int, optional): Number of input features, defaults to 3.
        scale (float, optional): Scaling factor for the output, defaults to 1.7e3.

    Methods:
        forward(inputs): Computes the scaled MLP output.

    Inputs:
        inputs (torch.Tensor): Input tensor of shape (N, input_features).

    Returns:
        torch.Tensor: Predicted values of shape (N, 1).
    """
    def __init__(self, input_features=3, u_size=64):
        super().__init__()
        self.size = u_size
        self.mlp = BuildMLP2(
            input_features=input_features,
            layers=[self.size, self.size, self.size, 1],
            activation=nn.SiLU(),
            linear_output=False,
            output_activation=nn.Softplus(),
            seed=0
        )

    def forward(self, inputs):
        # add dimensionality using self.scale
        return self.mlp(inputs)     


class Denoise(nn.Module):
    """
    Biologically-Informed Neural Network (BINN) for modeling cell density, diffusion, and growth.

    This class implements a neural network that combines cell density-dependent diffusion and growth MLPs 
    with an optional time delay MLP. It supports modeling and loss computation for biologically informed simulations.

    Attributes:
        surface_fitter (u_MLP): Multi-Layer Perceptron (MLP) for modeling cell density.
    
        x1_arr (ndarray): Array of x-coordinates for the domain.
        x2_arr (ndarray): Array of y-coordinates for the domain (optional for 2D).
        t_arr (ndarray): Array of time points for the domain.
        x1_min (float): Minimum x-coordinate value.
        x1_max (float): Maximum x-coordinate value.
        x2_min (float or None): Minimum y-coordinate value (if x2_arr is provided).
        x2_max (float or None): Maximum y-coordinate value (if x2_arr is provided).
        t_min (float): Minimum time value.
        t_max (float): Maximum time value.
        IC_weight (float): Weight for initial condition in loss computation.
        surface_weight (float): Weight for the surface fitting loss term.
        gamma (float): Proportionality constant for GLS weighting.
        num_samples (int): Number of samples used for PDE loss computation.
        num_features (int): Number of input features (2 for 1D, 3 for 2D).
        epochs (int): Number of training epochs completed.
        val_batch_it (int): Validation batch iteration count.
        tr_batch_it (int): Training batch iteration count.
        loss_count (int): Total loss computations performed.

    Args:
        x1_arr (ndarray): Array of x-coordinates.
        x2_arr (ndarray): Array of y-coordinates (optional for 2D).
        t_arr (ndarray): Array of time points.
        growth (bool, optional): Whether to include a growth MLP. Defaults to True.
        delay (bool, optional): Whether to include a time delay MLP. Defaults to False.

    Methods:
        __init__(self, x1_arr, x2_arr, t_arr, growth=True, delay=False):
            Initializes the BINN model with specified configurations.

        forward(self, inputs):
            Computes surface predictions using the cell density MLP.

        gls_loss(self, pred, true):
            Computes the Generalized Least Squares (GLS) loss for predictions.

        loss(self, pred, true):
            Computes the total loss combining GLS and PDE losses.
    """

    def __init__(self, 
                 data_obj,
                 model_params,
                 data_loss_func):
        super().__init__()


        # DATA
        # ----------------------------------------------------------------------------------------------
        x1 =  data_obj.x1
        x2 =  data_obj.x2
        t_arr =  data_obj.t
        K = data_obj.K
        u_clean =  data_obj.u_clean
        gamma =  data_obj.gamma
        u_min = np.min(u_clean)
        u_max = np.max(u_clean)
        
        # MODEL
        # ----------------------------------------------------------------------------------------------
        denoise_model_params = model_params["denoise_model_params"]
        # ----------------------------------------------------------------------------------------------
        denoise_construction_params = denoise_model_params["denoise_construction_params"] 
        device = denoise_construction_params['denoiseDevice']
        denoiseUsize = denoise_construction_params["denoiseUsize"]


        self.data_loss_func = data_loss_func

        # Initialize surface fitter and components
        self.surface_fitter = u_MLP_dn(u_size=denoiseUsize).to(device)
        
        self.x1 = x1
        self.x2 = x2
        self.t_arr = t_arr

        X1, X2, T = np.meshgrid(x1, x2, t_arr, indexing='ij')  # shape (len(x1), len(x2), len(t))
        self.meshgrid = np.column_stack([X1.ravel(), X2.ravel(), T.ravel()])
        self.torch_meshgrid = torch.tensor(self.meshgrid, dtype=torch.float32, device=device)

        self.x1_min = min(x1)
        self.x1_max = max(x1)
        self.x2_min = min(x2)
        self.x2_max = max(x2)
        self.t_min = min(t_arr)
        self.t_max = max(t_arr)
        
        self.u_scale = K

        #self.sigma_estimates = []
        #self.log_sigma = nn.Parameter(torch.tensor(0.0))  # log(sigma)

        # Loss weights
        self.IC_weight = 1e0
        self.surface_weight = 1e0
        
        self.gamma = gamma

        self.num_features = 2

        # Training tracking
        self.epochs = 1
        self.val_batch_it = 0
        self.tr_batch_it = 0
        self.loss_count = 0


    def forward(self, inputs):
        """
        Forward pass for the model.

        Inputs:
            inputs (torch.Tensor): Input tensor for surface prediction.

        Returns:
            torch.Tensor: Predicted cell densities.
        """
        self.inputs = inputs
        return self.surface_fitter(self.inputs)


    def loss(self, pred, true):
        """
        Computes the total loss combining GLS and PDE losses.

        Inputs:
            pred (torch.Tensor): Predicted values.
            true (torch.Tensor): True values.

        Returns:
            tuple: Total loss, GLS loss, and PDE loss values.
        """
        
        self.data_loss_val = 0
        self.data_loss_val_total = 0
        self.data_loss_val_total = self.data_loss_func(self, pred, true)

        # Compute mean GLS loss
        self.data_loss_val = self.surface_weight * torch.mean(self.data_loss_val_total)   
        self.loss_count += 1

        return (self.data_loss_val, self.data_loss_val_total)
    


