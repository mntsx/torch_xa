# python 3.12

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials


class AccumulateGradX(ExtendedAutogradFunction):

    def __init__(self, grad_fn: AutogradFunction, order: int) -> None:
        super().__init__(grad_fn=grad_fn, order=order)
        return None

    def integral(self) -> bool:
        return True

    def _get_context(self) -> None:
        return None

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        assert len(shaped_output_partials[0]) == self._order

        multipartials: list[list[Tensor]] = [list(shaped_output_partials[0])]
        shapes: list[list[Tensor]] = [shaped_output_partials[1]]
        self._update_multipartials(multipartials=multipartials, shapes=shapes)

        return None
