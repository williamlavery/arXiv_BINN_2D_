#print("""
#MODULE| `BuildMLP`| version = 19 Dec 24.

#Info:
#- Version general for all of `BINN` repository.
#- Sufficient module documentation.
#""")


import torch
import torch.nn as nn


class BuildMLP(nn.Module):
    """
    Builds a standard multilayer perceptron (MLP) with configurable options for activation functions,
    batch normalization, dropout, and output layer characteristics.

    This class supports custom initialization, optional batch normalization, and flexible activation 
    functions for hidden and output layers. It also allows setting a random seed for reproducibility.

    Attributes:
        input_features (int): Number of input features for the MLP.
        layers (list[int]): List of integers specifying the size of each layer.
        activation (nn.Module): Activation function for hidden layers. Defaults to nn.Sigmoid().
        linear_output (bool): If True, the output layer has no activation. Defaults to True.
        output_activation (nn.Module): Activation function for the output layer. Defaults to the same 
                                        as the hidden layer activation unless specified.
        use_batchnorm (bool): If True, batch normalization is applied to hidden layers. Defaults to False.
        dropout_rate (float): Dropout rate applied to hidden layers. Defaults to 0.0 (no dropout).
        seed (int): Random seed for reproducibility. Defaults to 0.

    Args:
        input_features (int): Number of input features.
        layers (list[int]): List of integers specifying the size of each layer.
        activation (nn.Module, optional): Activation function for hidden layers. Defaults to nn.Sigmoid().
        linear_output (bool, optional): If True, output layer is linear. Defaults to True.
        output_activation (nn.Module, optional): Activation function for the output layer. Defaults to None.
        use_batchnorm (bool, optional): If True, applies batch normalization to hidden layers. Defaults to False.
        dropout_rate (float, optional): Dropout rate for hidden layers. Defaults to 0.0.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Methods:
        forward(x):
            Computes the forward pass of the MLP.

    Inputs:
        x (torch.Tensor): Input tensor of shape (batch_size, input_features).

    Returns:
        torch.Tensor: Output tensor of shape (batch_size, layers[-1]).
    """
    
    def __init__(self, 
                 input_features, 
                 layers, 
                 activation=None, 
                 linear_output=True,
                 output_activation=None,
                 use_batchnorm=False,
                 dropout_rate=0.0,
                 seed=0):
        """
        Initializes the BuildMLP class with the specified configurations.

        Args:
            input_features (int): Number of input features.
            layers (list[int]): List of integers specifying the size of each layer.
            activation (nn.Module, optional): Activation function for hidden layers. Defaults to nn.Sigmoid().
            linear_output (bool, optional): If True, output layer is linear. Defaults to True.
            output_activation (nn.Module, optional): Activation function for the output layer. Defaults to None.
            use_batchnorm (bool, optional): If True, applies batch normalization to hidden layers. Defaults to False.
            dropout_rate (float, optional): Dropout rate for hidden layers. Defaults to 0.0.
            seed (int, optional): Random seed for reproducibility. Defaults to 0.
        """
        super().__init__()
        
        self.input_features = input_features
        self.layers = layers
        self.seed = seed
        self.activation = activation if activation is not None else nn.Sigmoid()
        self.linear_output = linear_output
        self.output_activation = output_activation if output_activation else self.activation
        self.use_batchnorm = use_batchnorm
        self.dropout_rate = dropout_rate

        # Set the random seed for reproducibility
        if self.seed is not None:
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)

        # Build the MLP
        operations = []
        for i, layer in enumerate(layers[:-1]):
            # Add linear layer
            operations.append(nn.Linear(in_features=self.input_features, out_features=layer, bias=True))
            self.input_features = layer
            
            # Add batch normalization (if enabled)
            if self.use_batchnorm:
                operations.append(nn.BatchNorm1d(layer))
            
            # Add activation function
            operations.append(self.activation)
            
            # Add dropout (if enabled)
            if self.dropout_rate > 0:
                operations.append(nn.Dropout(p=self.dropout_rate))
        
        # Add the final output layer
        operations.append(nn.Linear(in_features=self.input_features, out_features=layers[-1], bias=True))
        if not self.linear_output:
            operations.append(self.output_activation)
        
        # Convert the operations list into a sequential model
        self.MLP = nn.Sequential(*operations)

    def forward(self, x):
        """
        Computes the forward pass of the MLP.

        Inputs:
            x (torch.Tensor): Input tensor of shape (batch_size, input_features).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, layers[-1]).
        """
        return self.MLP(x)

        