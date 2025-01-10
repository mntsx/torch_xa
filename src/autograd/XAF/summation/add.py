# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class AddXBackward0(ExtendedAutogradFunction):

    def __init__(self, grad_fn: AutogradFunction, order: int) -> None:
        super().__init__(grad_fn=grad_fn, order=order)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[int, ...]:
        saved_alpha: float = self.grad_fn._saved_alpha
        return (saved_alpha,)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        alpha: float = self._get_context()[0]

        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        multipartials: list[list[Tensor]] = [[], []]
        shapes: list[Tuple[int, ...]] = [output_shape, output_shape]

        # compute input partials
        new_partial: Tensor
        for i, partial in enumerate(output_partials):
            if i == 0:
                new_partial = partial.clone()
            else:
                new_partial = torch.zeros_like(partial)
            multipartials[0].append(new_partial)

        # compute other partials
        for i, partial in enumerate(output_partials):
            if i == 0:
                new_partial = alpha * partial
            else:
                new_partial = torch.zeros_like(partial)
            multipartials[1].append(new_partial)

        self._update_multipartials(multipartials=multipartials, shapes=shapes)

        return None
