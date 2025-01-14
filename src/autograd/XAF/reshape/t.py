# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class TXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> None:
        return None

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        assert len(shaped_output_partials[0]) == self._order
        assert len(shaped_output_partials[1]) >= 2

        expected_output_shape: Tuple[int, ...] = (*shaped_output_partials[1][-2:],)
        shaped_output_partials = self._unbroadcast_partials(
            shaped_partials=shaped_output_partials, output_shape=expected_output_shape
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]

        multipartials: list[list[Tensor]] = [[]]
        multishapes: list[Tuple[int, ...]] = [tuple(output_shape[::-1])]

        for i, partial in enumerate(output_partials):

            # obtain reshape
            graph_output_numel: int = output_partials[0].shape[0]
            reshape: Tuple[int, ...] = (graph_output_numel,) + (i + 1) * output_shape

            # obtain full permutation
            pointer = 1
            perm_list: list[int] = [0]
            for _ in range(i + 1):
                perm_list.append(pointer + 1)
                perm_list.append(pointer)
                pointer += 2

            reshaped_partial: Tensor = partial.view(size=reshape)
            permuted_partial: Tensor = reshaped_partial.permute(dims=tuple(perm_list))
            shape: Tuple[int, ...] = tuple(partial.shape)
            unshaped_partial: Tensor = permuted_partial.reshape(shape=shape)
            unshaped_partial = unshaped_partial.contiguous()

            multipartials[0].append(unshaped_partial)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
