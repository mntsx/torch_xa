# python 3.12

# Standard Library dependencies
from typing import Union

# PyTorch dependencies
import torch
from torch import Tensor


def construct_nd_identity(
    n: int, dims: int, dtype: torch.dtype = torch.float32
) -> Tensor:
    """
    Constructs an n^dims-sized identity shaped as [n, n, ..., n] (dims times).
    For example, construct_nd_identity(2, 2) -> shape [2,2].
    """
    size: int = n**dims
    IDn: Tensor = torch.zeros((size,), dtype=dtype)
    idx: Tensor = torch.arange(0, n)
    factor: int = 0
    for i in range(dims):
        factor += n**i
    idx *= factor
    IDn[idx] = 1

    return IDn.view(*([n] * dims))


def diagonalize(flat_tensor: Tensor, dims: int) -> Tensor:

    n: int = flat_tensor.numel()
    size: int = n**dims
    IDn: Tensor = torch.zeros((size,), dtype=flat_tensor.dtype)
    idx: Tensor = torch.arange(0, n)
    factor: int = 0
    for i in range(dims):
        factor += n**i
    idx *= factor
    IDn[idx] = flat_tensor

    return IDn.view(*([n] * dims))


def einsum(*args: list[Union[list[int], Tensor]]) -> Tensor:
    tensor: Tensor = torch.einsum(*args)
    return tensor
