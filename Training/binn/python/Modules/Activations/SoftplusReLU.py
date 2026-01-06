# Version 19 Dec 24

import torch
import torch.nn as nn

class SoftplusReLU(nn.Module):
    """
    A hybrid activation function combining Softplus and ReLU to mitigate 
    floating-point issues caused by excessively large input values. For inputs 
    below the threshold, Softplus activation is applied. For inputs equal to or 
    exceeding the threshold, ReLU activation is used.

    Args:
        threshold (float): The cutoff value that determines whether to apply 
            Softplus or ReLU. Defaults to 20.0.

    Inputs:
        x (torch.Tensor): Input tensor containing floating-point values.

    Returns:
        torch.Tensor: Output tensor after applying the activation function.
    """

    def __init__(self, threshold=20.0):
        super(SoftplusReLU, self).__init__()
        self.threshold = threshold
        self.softplus = nn.Softplus()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the SoftplusReLU activation function.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Activated output tensor.
        """
        return torch.where(x < self.threshold, self.softplus(x), self.relu(x))

