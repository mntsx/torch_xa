# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop import diagonal_contraction
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unbroadcast
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class MulXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor, Tensor]:
        saved_other: Tensor = self.grad_fn._saved_other
        saved_self: Tensor = self.grad_fn._saved_self
        return (saved_other, saved_self)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0

        ctx: Tuple[Tensor, Tensor] = self._get_context()
        m2: Tensor = ctx[0]
        m1: Tensor = ctx[1]

        max_len: int = max(m1.ndim, m2.ndim)
        m1_padded_shape: Tuple[int, ...] = (1,) * (max_len - m1.ndim) + tuple(m1.shape)
        m2_padded_shape: Tuple[int, ...] = (1,) * (max_len - m2.ndim) + tuple(m2.shape)
        expected_output_shape: list[int] = list()
        for i in range(max_len):
            expected_output_shape.append(max(m1_padded_shape[i], m2_padded_shape[i]))
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials,
            output_shape=tuple(expected_output_shape),
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        derivative: Tensor
        derivatives: list[Tensor] = []
        multipartials: list[list[Tensor]] = [[], []]
        multishapes: list[Tuple[int, ...]] = [tuple(m1.shape), tuple(m2.shape)]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # compute self internal partials
        derivatives = list()
        broadcasted_m2: Tensor = m2.broadcast_to(size=output_shape)
        for order in range(1, self._order + 1):
            if order == 1:
                derivative = broadcasted_m2.flatten()
            else:
                derivative = torch.zeros(
                    size=(broadcasted_m2.numel(),), device=self._device
                )
            derivatives.append(derivative)

        # compute self partials
        pretensors: Tuple[Tensor, ...]
        subtensors: Tuple[Tensor, ...]
        pretensors = output_partials
        subtensors = tuple(derivatives)
        for expression in expressions:
            contracted_tensor: Tensor = diagonal_contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
            )
            multipartials[0].append(contracted_tensor)
        multipartials[0] = list(
            unbroadcast(
                shaped_partials=(multipartials[0], output_shape), output_shape=m1.shape
            )[0]
        )

        # compute other internal partials
        derivatives = list()
        broadcasted_m1: Tensor = m1.broadcast_to(size=output_shape)
        for order in range(1, self._order + 1):
            if order == 1:
                derivative = broadcasted_m1.flatten()
            else:
                derivative = torch.zeros(
                    size=(broadcasted_m1.numel(),), device=self._device
                )
            derivatives.append(derivative)

        # compute other partials
        pretensors = output_partials
        subtensors = tuple(derivatives)
        for expression in expressions:
            contracted_tensor = diagonal_contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
            )
            multipartials[1].append(contracted_tensor)
        multipartials[1] = list(
            unbroadcast(
                shaped_partials=(multipartials[1], output_shape), output_shape=m2.shape
            )[0]
        )

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
