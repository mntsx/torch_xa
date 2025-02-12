# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.engine.backprop import diagonal_contraction
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unbroadcast
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class PreluKernelXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor, Tensor]:
        saved_weight: Tensor = self._grad_fn._saved_weight
        saved_self: Tensor = self._grad_fn._saved_self
        return (saved_weight, saved_self)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0
        ctx: Tuple[Tensor, Tensor] = self._get_context()
        weight: Tensor = ctx[0]
        input: Tensor = ctx[1]

        expected_output_shape: Tuple[int, ...] = tuple(input.shape)
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials, output_shape=expected_output_shape
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order
        assert len(output_shape) > 1
        squeezed_weight: Tensor = weight.squeeze()
        assert squeezed_weight.ndim == 1
        assert squeezed_weight.shape[0] in (1, output_shape[1])

        multipartials: list[list[Tensor]] = [[], []]
        multishapes: list[Tuple[int, ...]] = [output_shape, output_shape]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=(n + 1)) for n in range(self._order)]

        # precalculations
        derivative: Tensor
        derivatives: list[Tensor]
        cond: Tensor = input.flatten() > 0
        t1: Tensor = torch.ones(size=(1,), device=self._device)
        t0: Tensor = torch.zeros(size=(1,), device=self._device)
        extend_shape: Tuple[int, ...]
        extend_shape = tuple([s if d == 1 else 1 for d, s in enumerate(output_shape)])
        extended_weight: Tensor = squeezed_weight.view(size=extend_shape)
        expanded_weight: Tensor = extended_weight.expand(size=output_shape)
        flat_input: Tensor = torch.flatten(input)
        flat_weight: Tensor = torch.flatten(expanded_weight)

        # Compute input internal partials
        derivatives = list()
        derivative: Tensor = torch.where(condition=cond, input=t1, other=flat_weight)
        for order in range(1, self._order + 1):
            if order > 1:
                # derivative = torch.zeros_like(derivative)
                derivative = torch.zeros(size=(1,), device=self._device)
            derivatives.append(derivative)

        # compute input partials
        pretensors: Tuple[Tensor, ...] = output_partials
        subtensors: Tuple[Tensor, ...] = tuple(derivatives)
        for expression in expressions:
            contracted_tensor: Tensor = diagonal_contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
            )
            multipartials[0].append(contracted_tensor)

        # Compute weight internal partials
        derivatives: list[Tensor] = list()
        derivative = torch.where(condition=cond, input=t0, other=flat_input)
        for order in range(1, self._order + 1):
            if order > 1:
                derivative = torch.zeros_like(derivative)
            derivatives.append(derivative)

        # compute weight partials
        pretensors: Tuple[Tensor, ...] = output_partials
        subtensors: Tuple[Tensor, ...] = tuple(derivatives)
        for expression in expressions:
            contracted_tensor: Tensor = diagonal_contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
            )
            multipartials[1].append(contracted_tensor)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
