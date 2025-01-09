# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop.contraction import contractor
from src.autograd.engine.backprop.derivation import calculate_n_order_partial, SumGroup
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class MmXBackward0(ExtendedAutogradFunction):

    def __init__(self, grad_fn: AutogradFunction, order: int) -> None:
        super().__init__(grad_fn=grad_fn, order=order)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(
        self,
    ) -> Tuple[
        Tensor,
        Tuple[int, ...],
        Tuple[int, ...],
        Tensor,
        Tuple[int, ...],
        Tuple[int, ...],
    ]:
        m1: Tensor = self._grad_fn._saved_self
        m1_sizes: Tuple[int, ...] = self.grad_fn._saved_self_sym_sizes
        m1_strides: Tuple[int, ...] = self.grad_fn._saved_self_sym_strides

        m2: Tensor = self._grad_fn._saved_mat2
        m2_sizes: Tuple[int, ...] = self.grad_fn._saved_mat2_sym_sizes
        m2_strides: Tuple[int, ...] = self.grad_fn._saved_mat2_sym_strides

        return (m1, m1_sizes, m1_strides, m2, m2_sizes, m2_strides)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0

        ctx: Tuple[Tensor, Tensor] = self._get_context()
        m1: Tensor = ctx[0]
        m1_sizes: Tuple[int, ...] = ctx[1]
        m2: Tensor = ctx[3]
        m2_sizes: Tuple[int, ...] = ctx[4]

        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]

        assert len(output_partials) == self._order
        assert output_shape == (m1_sizes[0], m2_sizes[1])

        multipartials: list[list[Tensor]] = [[], []]
        shapes: list[list[Tensor]] = [m1_sizes, m2_sizes]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # retrieve some data
        internal_partials: list[Tensor]
        size: Tuple[int, ...]
        graph_output_numel: int = output_partials[0].shape[0]

        # compute m1 internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            internal_partial: Tensor
            if order == 1:
                internal_partial = m2.T
            else:
                size = (m2_sizes[1], *[m2_sizes[0] for _ in range(order)])
                internal_partial = torch.zeros(size=size)
            internal_partials.append(internal_partial)

        # compute m1 partials
        pretensors: Tuple[Tensor, ...]
        subtensors: Tuple[Tensor, ...]
        aux: list[Tensor] = list()
        for i, partial in enumerate(output_partials):
            size = (partial.shape[0], *(list(output_shape) * (i + 1)))
            aux.append(partial.view(size=size))
        pretensors = tuple(aux)
        subtensors = tuple(internal_partials)
        contracted_tensor: Tensor
        for i, expression in enumerate(expressions):
            contracted_tensor = contractor(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                batch=True,
            )
            dual_numel: int = m1_sizes[0] * m2_sizes[0]
            size = (graph_output_numel, *[dual_numel for _ in range(i + 1)])
            multipartials[0].append(contracted_tensor.view(size=size))

        # compute m2 internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            internal_partial: Tensor
            if order == 1:
                internal_partial = m1
            else:
                size: Tuple[int, ...]
                size = (m1_sizes[0], *[m1_sizes[1] for _ in range(order)])
                internal_partial = torch.zeros(size=size)
            internal_partials.append(internal_partial)

        # compute m2 partials
        aux: list[Tensor] = list()
        for i, pretensor in enumerate(pretensors):
            dims: list[int]
            dims = [j - 1 if j % 2 == 0 else j + 1 for j in range(1 + 2 * (i + 1))]
            dims[0] = 0
            aux.append(pretensor.permute(dims=tuple(dims)))
        pretensors = tuple(aux)
        subtensors = tuple(internal_partials)
        for i, expression in enumerate(expressions):
            contracted_tensor = contractor(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                batch=True,
            )
            dual_numel: int = m1_sizes[1] * m2_sizes[1]
            size = (graph_output_numel, *[dual_numel for _ in range(i + 1)])
            multipartials[1].append(contracted_tensor.view(size=size))

        self._update_multipartials(multipartials=multipartials, shapes=shapes)

        return None
