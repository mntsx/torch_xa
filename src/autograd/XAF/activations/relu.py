# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class ReluXBackward0(ExtendedAutogradFunction):

    def __init__(self, grad_fn: AutogradFunction, order: int) -> None:
        super().__init__(grad_fn=grad_fn, order=order)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor]:
        saved_result: Tensor = self._grad_fn._saved_result
        return (saved_result,)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0
        ctx: Tuple[Tensor, Tensor] = self._get_context()
        result: Tensor = ctx[0]

        expected_output_shape: Tuple[int, ...] = tuple(result.shape)
        shaped_output_partials = self._unbroadcast_partials(
            shaped_partials=shaped_output_partials,
            output_shape=expected_output_shape,
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        multipartials: list[list[Tensor]] = [[]]
        shapes: list[Tuple[int, ...]] = [output_shape]

        # compute partials
        new_partial: Tensor
        for i, partial in enumerate(output_partials):
            if i == 0:
                # obtain element wise internal first derivative tensor
                cond: Tensor = result.flatten() > 0
                t1: Tensor = torch.tensor([1.0])
                t0: Tensor = torch.tensor([0.0])
                derivative: Tensor = torch.where(condition=cond, input=t1, other=t0)
                # apply the internal derivative element wise over the output partial
                new_partial = partial * derivative.unsqueeze(0)
            else:
                new_partial = torch.zeros_like(partial)
            multipartials[0].append(new_partial)

        self._update_multipartials(multipartials=multipartials, shapes=shapes)

        return None
