from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

def vicreg_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    inv_coeff: float = 25.0,
    var_coeff: float = 15.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
):
    """Computes the VICReg loss.

    ---
    Args:
        x: Features map.
            Shape of [batch_size, representation_size].
        y: Features map.
            Shape of [batch_size, representation_size].
        inv_coeff: Coefficient for the invariance loss.
        var_coeff: Coefficient for the variance loss.
        cov_coeff: Coefficient for the covariance loss.
        gamma: Threshold for the variance loss.

    ---
    Returns:
        A dictionary containing:
            - "total_loss": The total VICReg loss.
            - "inv_loss": The invariance loss.
            - "var_loss": The variance loss.
            - "cov_loss": The covariance loss.
    """
    inv_loss = inv_coeff * representation_loss(x, y)
    var_loss = var_coeff * (
        variance_loss(x, gamma) + variance_loss(y, gamma)
    ) / 2
    cov_loss = cov_coeff * (covariance_loss(x) + covariance_loss(y)) / 2
    total_loss = inv_loss + var_loss + cov_loss

    return total_loss, inv_loss, var_loss, cov_loss


def representation_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Computes the representation loss.
    Force the representations of the same object to be similar.

    ---
    Args:
        x: Features map.
            Shape of [batch_size, representation_size].
        y: Features map.
            Shape of [batch_size, representation_size].

    ---
    Returns:
        The representation loss.
            Shape of [1,].
    """
    return F.mse_loss(x, y)


def variance_loss(x: torch.Tensor, gamma: float) -> torch.Tensor:
    """Computes the variance loss.
    Push the representations across the batch
    to be different between each other.
    Avoid the model to collapse to a single point.

    The gamma parameter is used as a threshold so that
    the model is no longer penalized if its std is above
    that threshold.

    ---
    Args:
        x: Features map.
            Shape of [batch_size, representation_size].

    ---
    Returns:
        The variance loss.
            Shape of [1,].
    """
    x = x - x.mean(dim=0)
    std = x.std(dim=0)
    var_loss = F.relu(gamma - std).mean()
    return var_loss


def covariance_loss(x: torch.Tensor) -> torch.Tensor:
    """Computes the covariance loss.
    Decorrelates the embeddings' dimensions, which pushes
    the model to capture more information per dimension.

    ---
    Args:
        x: Features map.
            Shape of [batch_size, representation_size].

    ---
    Returns:
        The covariance loss.
            Shape of [1,].
    """
    x = x - x.mean(dim=0)
    cov = (x.T @ x) / (x.shape[0] - 1)
    cov_loss = cov.fill_diagonal_(0.0).pow(2).sum() / x.shape[1]
    return cov_loss