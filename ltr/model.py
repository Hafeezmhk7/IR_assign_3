from torch import nn
from collections import OrderedDict
import torch.nn.functional as F
import torch


class LTRModel(nn.Module):
    """
    A simple Learning to Rank (LTR) model using a feedforward neural network.

    Attributes
    ----------
    layers : nn.Sequential
        A sequential container consisting of an input layer, a ReLU activation,
        and an output layer.
    """

    def __init__(self, num_features: int) -> None:
        """
        Initializes the LTR model.

        Parameters
        ----------
        num_features : int
            The number of input features.
        """
        super().__init__()

        ## BEGIN SOLUTION
        self.layers = nn.Sequential(
            nn.Linear(num_features, 10),  # Input layer to hidden layer
            nn.ReLU(),                    # ReLU activation
            nn.Linear(10, 1))
        
        ## END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (N, num_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (N, 1).
        """
        ## BEGIN SOLUTION
        return self.layers(x)

        ## END SOLUTION


class CLTRModel(nn.Module):
    """
    A Counterfactual Learning to Rank (CLTR) model with a tunable hidden layer width.

    Attributes
    ----------
    layers : nn.Sequential
        A sequential container consisting of an input layer, a ReLU activation,
        and an output layer.
    """

    def __init__(self, num_features: int, width: int) -> None:
        """
        Initializes the Counterfactual LTR model.

        Parameters
        ----------
        num_features : int
            The number of input features.
        width : int
            The number of hidden units in the intermediate layer.
        """
        super().__init__()

        ## BEGIN SOLUTION
        self.layers = nn.Sequential(
        nn.Linear(num_features, width),  # Input layer to hidden layer with configurable width
        nn.ReLU(),                       # ReLU activation
        nn.Linear(width, 1)              # Hidden layer to output layer
        )
        ## END SOLUTION



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (1, N, num_features).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (1, N, 1).
        """
        # BEGIN SOLUTION
        # Check input dimensionality and handle accordingly
        
        if x.dim() == 3:
            # Case 1: 3D input [batch_size, num_docs, num_features]
            batch_size, num_docs, num_features = x.shape
            x_reshaped = x.view(-1, num_features)
            output = self.layers(x_reshaped)
            return output.view(batch_size, num_docs, 1)
        elif x.dim() == 2:
            # Case 2: 2D input [batch_size, num_features] during evaluation
            output = self.layers(x)
            # Return without reshaping for evaluation
            return output
        else:
            # Unexpected shape
            raise ValueError(f"Unexpected input shape: {x.shape}. Expected either 2D or 3D tensor.")
        # END SOLUTION
