# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unbroadcast
from src.utils.types import AutogradFunction, ShapedPartials


class AccumulateGradX(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        return True

    def _get_context(self) -> Tuple[Tensor]:
        variable: Tensor = self._grad_fn.variable
        return (variable,)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        assert len(shaped_output_partials[0]) == self._order

        variable: Tensor = self._get_context()[0]

        expected_output_shape: Tuple[int, ...] = tuple(variable.shape)
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials,
            output_shape=expected_output_shape,
        )

        multipartials: list[list[Tensor]] = [list(shaped_output_partials[0])]
        multishapes: list[Tuple[int, ...]] = [shaped_output_partials[1]]
        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
