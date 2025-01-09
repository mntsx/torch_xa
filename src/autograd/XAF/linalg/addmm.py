# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop.contractions import contractor
from src.autograd.engine.backprop.derivation import calculate_n_order_partial, SumGroup
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class AddmmXBackward0(ExtendedAutogradFunction):

    def __init__(self, grad_fn: AutogradFunction, order: int) -> None:
        super().__init__(grad_fn=grad_fn, order=order)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(
        self,
    ) -> Tuple[
        float,
        float,
        Tensor,
        Tuple[int, ...],
        Tuple[int, ...],
        Tensor,
        Tuple[int, ...],
        Tuple[int, ...],
    ]:
        alpha: float = self._grad_fn._saved_alpha
        beta: float = self._grad_fn._saved_beta

        m1: Tensor = self._grad_fn._saved_mat1
        m1_sizes: Tuple[int, ...] = self.grad_fn._saved_mat1_sym_sizes
        m1_strides: Tuple[int, ...] = self.grad_fn._saved_mat1_sym_strides

        m2: Tensor = self._grad_fn._saved_mat2
        m2_sizes: Tuple[int, ...] = self.grad_fn._saved_mat2_sym_sizes
        m2_strides: Tuple[int, ...] = self.grad_fn._saved_mat2_sym_strides

        return (alpha, beta, m1, m1_sizes, m1_strides, m2, m2_sizes, m2_strides)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0
        ctx: Tuple[Tensor, Tensor] = self._get_context()

        alpha: float = ctx[0]
        beta: float = ctx[1]
        m1: Tensor = ctx[2]  # input
        m1_sizes: Tuple[int, ...] = ctx[3]
        m1_strides: Tuple[int, ...] = ctx[4]
        m2: Tensor = ctx[5]  # param.T
        m2_sizes: Tuple[int, ...] = ctx[6]
        m2_strides: Tuple[int, ...] = ctx[7]

        raise NotImplementedError("AddmmXBackward0 is not implemented.")

        partials_list: list[Tensor, ...]
        numel: int = tensor.numel()
        for i in range(order):
            if i == 0:
                partial: Tensor = beta * mat2.T
            else:
                partial: Tensor = torch.zeros(size=(numel for _ in range(i + 2)))
            partials_list.append(partial)

        partials: Tuple[Tuple[Tensor, ...], int] = ((tuple(partials_list), 0),)

        return None  # (input_partials, mat1_partials, mat2_partials)
