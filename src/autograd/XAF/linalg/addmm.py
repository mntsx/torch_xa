# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop import contraction, diagonal_contraction
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unbroadcast
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class AddmmXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
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

        m1: Tensor = self._grad_fn._saved_mat1.to(device=self._device)
        m1_sizes: Tuple[int, ...] = self._grad_fn._saved_mat1_sym_sizes
        m1_strides: Tuple[int, ...] = self._grad_fn._saved_mat1_sym_strides

        m2: Tensor = self._grad_fn._saved_mat2.to(device=self._device)
        m2_sizes: Tuple[int, ...] = self._grad_fn._saved_mat2_sym_sizes
        m2_strides: Tuple[int, ...] = self._grad_fn._saved_mat2_sym_strides

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
        m2: Tensor = ctx[5]  # param.T
        m2_sizes: Tuple[int, ...] = ctx[6]

        expected_output_shape: Tuple[int, ...] = (m1_sizes[0], m2_sizes[1])
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials, output_shape=expected_output_shape
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        multipartials: list[list[Tensor]] = [[], [], []]
        multishapes: list[Tuple[int, ...]] = [
            (m1_sizes[0], m2_sizes[1]),
            m1_sizes,
            m2_sizes,
        ]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # retrieve some data
        internal_partial: Tensor
        internal_partials: list[Tensor] = []
        shape: Tuple[int, ...]
        output_numel: int = m1_sizes[0] * m2_sizes[1]
        graph_output_numel: int = output_partials[0].shape[0]

        # compute input internal partials
        internal_partials = list()
        for i, partial in enumerate(output_partials):
            if i == 0:
                internal_partial = alpha * torch.ones(
                    size=(output_numel,), device=self._device
                )
            else:
                internal_partial = torch.zeros(
                    size=(output_numel,), device=self._device
                )
            internal_partials.append(internal_partial)

        # compute input partials
        pretensors: Tuple[Tensor, ...]
        subtensors: Tuple[Tensor, ...]
        aux: list[Tensor] = list()
        pretensors = output_partials
        subtensors = tuple(internal_partials)
        for expression in expressions:
            contracted_tensor = diagonal_contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
            )
            multipartials[0].append(contracted_tensor)

        # compute m1 internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            if order == 1:
                internal_partial = beta * m2.T
            else:
                shape = (m2_sizes[1], *[m2_sizes[0] for _ in range(order)])
                internal_partial = torch.zeros(size=shape, device=self._device)
            internal_partials.append(internal_partial)

        # compute m1 partials
        aux = list()
        for i, partial in enumerate(output_partials):
            shape = (graph_output_numel, *(list(output_shape) * (i + 1)))
            aux.append(partial.view(size=shape))
        pretensors = tuple(aux)
        subtensors = tuple(internal_partials)
        for i, expression in enumerate(expressions):
            contracted_tensor = contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                batch=(True, False),
                device=self._device,
            )
            dual_numel: int = m1_sizes[0] * m2_sizes[0]
            shape = (graph_output_numel, *[dual_numel for _ in range(i + 1)])
            multipartials[1].append(contracted_tensor.reshape(shape=shape))

        # compute m2 internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            if order == 1:
                internal_partial = beta * m1
            else:
                shape: Tuple[int, ...]
                shape = (m1_sizes[0], *[m1_sizes[1] for _ in range(order)])
                internal_partial = torch.zeros(size=shape, device=self._device)
            internal_partials.append(internal_partial)

        # compute m2 partials
        aux = list()
        for i, pretensor in enumerate(pretensors):
            dims: list[int]
            dims = [j - 1 if j % 2 == 0 else j + 1 for j in range(1 + 2 * (i + 1))]
            dims[0] = 0
            aux.append(pretensor.permute(dims=tuple(dims)))
        pretensors = tuple(aux)
        subtensors = tuple(internal_partials)
        for i, expression in enumerate(expressions):
            contracted_tensor: Tensor = contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                batch=(True, False),
                device=self._device,
            )
            dual_numel: int = m1_sizes[1] * m2_sizes[1]
            shape = (graph_output_numel, *[dual_numel for _ in range(i + 1)])
            multipartials[2].append(contracted_tensor.reshape(shape=shape))

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
