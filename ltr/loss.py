from torch.nn import functional as F
import torch
import itertools
import pandas as pd
import numpy as np


def pointwise_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean squared error (MSE) regression loss.

    Parameters
    ----------
    output : torch.Tensor
        Predicted values of shape [N, 1].
    target : torch.Tensor
        Ground truth values of shape [N].

    Returns
    -------
    torch.Tensor
        Scalar loss value.
    """
    assert target.dim() == 1
    assert output.size(0) == target.size(0)
    assert output.size(1) == 1

    ## BEGIN SOLUTION
    ## END SOLUTION


def pairwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the pairwise loss for a single query.

    The loss is calculated for all possible orderings in a query using sigma=1.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1], where N is the number of <query, document> pairs.
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor or None
        Mean pairwise loss if N >= 2, otherwise None.
    """
    if labels.size(0) < 2:
        return None

    ## BEGIN SOLUTION
    ## END SOLUTION


def compute_lambda_i(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes lambda_i using the LambdaRank approach (sigma=1).

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1], where N is the number of <query, document> pairs.
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor
        Lambda updates of shape [N, 1].
    """

    ## BEGIN SOLUTION
    ## END SOLUTION


def mean_lambda(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean and squared mean of LambdaRank updates.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1].
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor
        Tensor containing mean and squared mean lambda values.
    """
    return torch.stack([
        compute_lambda_i(scores, labels).mean(),
        torch.square(compute_lambda_i(scores, labels)).mean(),
    ])


def listwise_loss(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the LambdaRank loss (sigma=1).

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1].
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor
        LambdaRank loss of shape [N, 1].
    """
    
    ## BEGIN SOLUTION
    ## END SOLUTION


def mean_lambda_list(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes the mean and squared mean of LambdaRank updates.

    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores of shape [N, 1].
    labels : torch.Tensor
        Relevance labels of shape [N].

    Returns
    -------
    torch.Tensor
        A tensor containing the mean and squared mean lambda values.
    """
    return torch.stack(
        [
            listwise_loss(scores, labels).mean(),
            torch.square(listwise_loss(scores, labels)).mean(),
        ]
    )


def listNet_loss(output: torch.Tensor, target: torch.Tensor, grading: bool = False) -> torch.Tensor:
    """
    Computes the ListNet loss, introduced in "Learning to Rank: From Pairwise Approach to Listwise Approach".

    This loss is based on the probability distributions of ranking scores and relevance labels.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions of shape [1, topk, 1].
    target : torch.Tensor
        Ground truth labels of shape [1, topk].
    grading : bool, optional
        If True, returns additional debugging information. Default is False.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        If `grading=False`, returns a single loss value as a tensor.
        If `grading=True`, returns a tuple containing the loss and additional debugging information.
    """
    eps = 1e-10  # Small epsilon value for numerical stability

    ## BEGIN SOLUTION
    ## END SOLUTION

    if grading:
        return loss, {
            "preds_smax": preds_smax,
            "true_smax": true_smax,
            "preds_log": preds_log,
        }
    else:
        return loss
    

def unbiased_listNet_loss(
    output: torch.Tensor, target: torch.Tensor, propensity: torch.Tensor, grading: bool = False
) -> torch.Tensor:
    """
    Computes the Unbiased ListNet loss, incorporating propensity scores for unbiased learning to rank.

    Parameters
    ----------
    output : torch.Tensor
        Model predictions of shape [1, topk, 1].
    target : torch.Tensor
        Ground truth labels of shape [1, topk].
    propensity : torch.Tensor
        Propensity scores of shape [1, topk] or [topk], used for debiasing.
    grading : bool, optional
        If True, returns additional debugging information. Default is False.

    Returns
    -------
    torch.Tensor or Tuple[torch.Tensor, Dict[str, torch.Tensor]]
        If `grading=False`, returns a single loss value as a tensor.
        If `grading=True`, returns a tuple containing the loss and additional debugging information.
    """
    eps = 1e-10  # Small epsilon value for numerical stability

    # Clip propensity scores to avoid division by small values, improving stability and lowering variance
    stable_propensity = propensity.clip(0.01, 1)

    ## BEGIN SOLUTION
    ## END SOLUTION

    if grading:
        return loss, {
            "preds_smax": preds_smax,
            "true_smax": true_smax,
            "preds_log": preds_log,
        }
    else:
        return loss
