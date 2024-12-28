import warnings
from typing import Iterable, Union, Tuple
import torch
from torch._six import inf
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_tensor_or_tensors = Union[torch.Tensor, Iterable[torch.Tensor]]


def clip_grad_norm_(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False
) -> torch.Tensor:
    """
    Clips gradient norm of an iterable of parameters. Supports dynamic threshold adjustment and distributed operations.

    Args:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a single Tensor that will have gradients normalized.
        max_norm (float): Maximum allowable norm of the gradients.
        norm_type (float): Type of the used p-norm. Can be `'inf'` for infinity norm.
        error_if_nonfinite (bool): If True, raise an error if the total norm is non-finite.

    Returns:
        torch.Tensor: Total norm of the parameters.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].device
    if norm_type == inf:
        norms = [p.detach().abs().max().to(device) for p in parameters if p.grad is not None]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in parameters if p.grad is not None]),
            norm_type
        )

    if total_norm.isnan() or total_norm.isinf():
        if error_if_nonfinite:
            raise RuntimeError(f"Non-finite norm encountered with norm type {norm_type}")
        else:
            warnings.warn("Non-finite norm encountered during gradient clipping. Continuing anyway.", RuntimeWarning)

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef.to(p.device))
                logger.debug(f"Clipped gradient for parameter on device {p.device}. Clip coefficient: {clip_coef:.4f}")

    logger.info(f"Gradient norm: {total_norm:.4f}, Max allowed norm: {max_norm:.4f}, Clip coefficient: {clip_coef:.4f}")
    return total_norm


def dynamic_clip_grad_norm(
    parameters: _tensor_or_tensors,
    initial_max_norm: float,
    adjustment_factor: float = 0.9,
    performance_metric: float = None,
    norm_type: float = 2.0,
    min_threshold: float = 1e-3,
    max_threshold: float = 1e3
) -> torch.Tensor:
    """
    Dynamically adjusts and clips the gradient norm based on performance metrics.

    Args:
        parameters (Iterable[Tensor] or Tensor): Parameters with gradients to clip.
        initial_max_norm (float): Initial maximum norm value.
        adjustment_factor (float): Factor by which to adjust max_norm based on performance.
        performance_metric (float): Metric to evaluate performance (e.g., loss value).
        norm_type (float): Norm type for clipping.
        min_threshold (float): Minimum allowable threshold for max_norm.
        max_threshold (float): Maximum allowable threshold for max_norm.

    Returns:
        torch.Tensor: Total norm of the parameters.
    """
    if performance_metric is not None:
        if performance_metric > 1.0:  # Example: Higher loss -> decrease max_norm
            initial_max_norm = max(initial_max_norm * adjustment_factor, min_threshold)
        else:  # Lower loss -> increase max_norm
            initial_max_norm = min(initial_max_norm / adjustment_factor, max_threshold)

        logger.info(f"Adjusted max_norm to {initial_max_norm:.4f} based on performance metric: {performance_metric:.4f}")

    return clip_grad_norm_(parameters, initial_max_norm, norm_type)


def distributed_clip_grad_norm(
    parameters: _tensor_or_tensors,
    max_norm: float,
    norm_type: float = 2.0,
    world_size: int = 1,
    error_if_nonfinite: bool = False
) -> torch.Tensor:
    """
    Clips gradient norm in a distributed training setup.

    Args:
        parameters (Iterable[Tensor] or Tensor): Parameters with gradients to clip.
        max_norm (float): Maximum norm value.
        norm_type (float): Norm type for clipping.
        world_size (int): Number of distributed workers.
        error_if_nonfinite (bool): Raise error if the total norm is non-finite.

    Returns:
        torch.Tensor: Total norm of the parameters.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    max_norm = float(max_norm)
    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].device
    if norm_type == inf:
        norms = [p.grad.detach().abs().max().to(device) for p in parameters if p.grad is not None]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters if p.grad is not None]),
            norm_type
        )

    if world_size > 1:
        # Reduce total_norm across all distributed workers
        torch.distributed.all_reduce(total_norm, op=torch.distributed.ReduceOp.SUM)
        total_norm = total_norm / world_size

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(clip_coef.to(p.device))
                logger.debug(f"Distributed gradient clipping applied. Clip coefficient: {clip_coef:.4f}")

    logger.info(f"Distributed Gradient Norm: {total_norm:.4f}, Max Norm: {max_norm:.4f}")
    return total_norm


def clip_grad_value_(
    parameters: _tensor_or_tensors,
    clip_value: float
) -> None:
    """
    Clips gradient values within a specified range.

    Args:
        parameters (Iterable[Tensor] or Tensor): Parameters with gradients to clip.
        clip_value (float): Maximum allowed absolute value of gradients.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        p.grad.data.clamp_(min=-clip_value, max=clip_value)
        logger.debug(f"Clipped gradient values for parameter on device {p.device} to range [-{clip_value}, {clip_value}].")
