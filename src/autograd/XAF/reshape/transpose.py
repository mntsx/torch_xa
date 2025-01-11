# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class TransposeXBackward0(ExtendedAutogradFunction):

    def __init__(self, grad_fn: AutogradFunction, order: int) -> None:
        super().__init__(grad_fn=grad_fn, order=order)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[int, int]:
        saved_dim0: int = self._grad_fn.saved_dim0
        saved_dim1: int = self._grad_fn.saved_dim1
        return (saved_dim0, saved_dim1)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0

        ctx: Tuple[int, int] = self._get_context()
        d0: int = ctx[0]
        d1: int = ctx[1]

        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order
        assert len(output_shape) >= max(d0, d1) + 1

        multipartials: list[list[Tensor]] = [[]]
        multishapes: list[Tuple[int, ...]] = []

        for i, partial in enumerate(output_partials):

            # obtain reshape
            graph_output_numel: int = output_partials[0].shape[0]
            reshape: Tuple[int, ...] = (graph_output_numel,) + (i + 1) * output_shape

            # obtain permutation
            permutation: list[int] = list(range(len(output_shape)))
            permutation[d0] = d1
            permutation[d1] = d0

            # obtain full permutation
            pointer = 1
            perm_list: list[int] = [0]
            for _ in range(i + 1):
                perm_list.extend([d + pointer for d in permutation])
                pointer += len(permutation)

            reshaped_partial: Tensor = partial.view(size=reshape)
            permuted_partial: Tensor = reshaped_partial.permute(dims=tuple(perm_list))
            unshaped_partial: Tensor = permuted_partial.view(size=tuple(partial.shape))
            unshaped_partial = unshaped_partial.contiguous()

            multishapes = [(*[output_shape[d] for d in permutation],)]
            multipartials[0].append(unshaped_partial)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
