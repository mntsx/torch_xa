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
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class WhereXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor]:
        saved_condition: float = self._grad_fn._saved_condition
        return (saved_condition,)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        condition: float = self._get_context()[0]

        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        derivative: Tensor
        derivatives: list[Tensor]
        multipartials: list[list[Tensor]] = [[], []]
        multishapes: list[Tuple[int, ...]] = [output_shape, output_shape]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # Tensors
        flat_cond: Tensor = torch.flatten(condition)
        t1: Tensor = torch.ones(size=(1,), device=self._device)
        t0: Tensor = torch.zeros(size=(1,), device=self._device)

        # compute other internal partials
        derivatives = list()
        derivative = torch.where(condition=flat_cond, input=t1, other=t0)
        for order in range(1, self._order + 1):
            if order > 1:
                # derivative = torch.zeros_like(input=derivative, device=self._device)
                derivative = torch.zeros(size=(1,), device=self._device)
            derivatives.append(derivative)

        # compute other partials
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

        # compute other internal partials
        derivatives = list()
        derivative = torch.where(condition=flat_cond, input=t0, other=t1)
        for order in range(1, self._order + 1):
            if order > 1:
                derivative = torch.zeros_like(input=derivative, device=self._device)
            derivatives.append(derivative)

        # compute other partials
        pretensors = output_partials
        subtensors = tuple(derivatives)
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
