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
        ## BEGIN SOLUTION
        ## END SOLUTION
