# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials


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
        raise NotImplementedError("TransposeXBackward0 is not implemented.")
        return None
