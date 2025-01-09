# python 3.12

# Standard Library dependencies
from typing import Optional, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.utils.types import Partials, ShapedPartials


def start_partials(tensor: Tensor, order: int) -> ShapedPartials:
    partials_list: list[Tensor] = []
    numel: int = tensor.numel()
    for i in range(order):
        if i == 0:
            partial: Tensor = torch.eye(numel)
        else:
            partial: Tensor = torch.zeros(size=tuple([numel for _ in range(i + 2)]))
        partials_list.append(partial)
    partials: Partials = tuple(partials_list)
    shape: Tuple[int, ...] = tuple(tensor.shape)
    shaped_partials: ShapedPartials = (partials, shape)
    return shaped_partials


def get_backward_idx(tensor: torch.Tensor) -> Optional[int]:
    dummy: Tensor = tensor.sum()
    idx: Union[None, int] = None
    # Check the position in next_functions where our tensor's grad_fn appears
    for grad_fn, pos in dummy.grad_fn.next_functions:
        if grad_fn is tensor.grad_fn:
            idx = pos
    dummy.detach()
    return idx


def sum_partials(partials_list: list[ShapedPartials]) -> ShapedPartials:
    assert len(set([len(p) for p in partials_list])) == 1
    sum_list: list[Tensor] = []
    for i, _ in enumerate(partials_list[0]):
        for j, _ in enumerate(partials_list):
            partial: Tensor
            if j == 0:
                partial = partials_list[0][0][i]
            else:
                partial += partials_list[j][0][i]
        sum_list.append(partial)
    partials: Partials = tuple(sum_list)
    shape: Tuple[int, ...] = partials_list[0][1]
    shaped_partials: ShapedPartials = (partials, shape)
    return shaped_partials
