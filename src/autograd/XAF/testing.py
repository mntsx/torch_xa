# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, Partials, ShapedPartials


class TestXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        return True

    def _get_context(self) -> None:
        return None

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        aux: list[Partials] = list()
        for i in range(10):  # assume no operator will take more thant 10 inputs
            partials: Partials = tuple([torch.tensor([i]) for _ in range(self._order)])
            shape: Tuple[int, ...] = (1,)
            shaped_partials: ShapedPartials = (partials, shape)
            aux.append(shaped_partials)
        multipartials = tuple(aux)
        self._multipartials: Tuple[Partials] = multipartials

        return None
