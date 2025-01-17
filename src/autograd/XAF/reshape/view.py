# python 3.12

# Standard Library dependencies
import math
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unscale
from src.utils.types import AutogradFunction, ShapedPartials


class ViewXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tuple[int, ...]]:
        saved_self_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_self_sym_sizes
        return (saved_self_sym_sizes,)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        assert len(shaped_output_partials[0]) == self._order

        self_sym_sizes: Tuple[int, ...] = self._get_context()[0]
        shaped_output_partials = unscale(
            shaped_partials=shaped_output_partials,
            target_numel=math.prod(self_sym_sizes),
        )

        multipartials: list[list[Tensor]] = [shaped_output_partials[0]]
        multishapes: list[Tuple[int, ...]] = [self_sym_sizes]

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
