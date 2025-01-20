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


class PowXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[float, Tensor]:
        saved_exponent: float = self._grad_fn._saved_exponent
        saved_self: Tensor = self._grad_fn._saved_self
        return (saved_exponent, saved_self)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0
        ctx: Tuple[float, Tensor] = self._get_context()
        exponent: float = ctx[0]
        input: Tensor = ctx[1]

        expected_output_shape: Tuple[int, ...] = tuple(input.shape)
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials, output_shape=expected_output_shape
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        multipartials: list[list[Tensor]] = [[]]
        multishapes: list[Tuple[int, ...]] = [output_shape]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=(n + 1)) for n in range(self._order)]

        # precalculations
        flat_input: Tensor = torch.flatten(input=input)
        coefficient: float = 1.0

        # obtain element wise internal first derivative tensor
        derivatives: list[Tensor] = list()
        for order in range(1, self._order + 1):
            derivative: Tensor
            if exponent > 0:
                derivative = coefficient * flat_input**exponent
                coefficient *= exponent
                exponent -= 1
            else:
                # derivative = torch.zeros_like(input=input)
                derivative = torch.zeros(size=(1,), device=self._device)
            derivatives.append(derivative)

        # compute partials
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

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None


class PowXBackward1(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor, Tensor, Tensor]:
        saved_exponent: Tensor = self._grad_fn._saved_exponent.to(device=self._device)
        saved_result: Tensor = self._grad_fn._saved_result.to(device=self._device)
        saved_self: Tensor = self._grad_fn._saved_self.to(device=self._device)
        return (saved_exponent, saved_result, saved_self)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0
        ctx: Tuple[Tensor, Tensor, Tensor] = self._get_context()
        exponent: Tensor = ctx[0]
        result: Tensor = ctx[1]
        input: Tensor = ctx[2]

        expected_output_shape: Tuple[int, ...] = tuple(input.shape)
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials, output_shape=expected_output_shape
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        multipartials: list[list[Tensor]] = [[], []]
        multishapes: list[Tuple[int, ...]] = [output_shape, tuple(exponent.shape)]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=(n + 1)) for n in range(self._order)]

        # precalculations
        derivative: Tensor
        derivatives: list[Tensor]
        coefficient: Tensor
        flat_input: Tensor = torch.flatten(input=input)
        flat_exp: Tensor = torch.flatten(input=exponent)
        flat_exp_cp: Tensor = flat_exp.clone().detach().to(device=self._device)
        ones: Tensor = torch.ones_like(input=flat_exp)

        # compute input internal partials
        derivatives = list()
        coefficient = ones
        for _ in range(1, self._order + 1):
            derivative: Tensor
            if flat_exp_cp.max() > 0:
                coefficient *= flat_exp_cp
                flat_exp_cp -= 1
                derivative = coefficient * flat_input**flat_exp_cp
            else:
                # derivative = torch.zeros_like(input=input)
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

        # precalculations
        log_input: Tensor = torch.log(input=flat_input)
        flat_exp = torch.flatten(input=exponent)
        flat_result: Tensor = torch.flatten(input=result)

        # compute input internal partials
        derivatives: list[Tensor] = list()
        coefficient = ones
        for order in range(1, self._order + 1):
            coefficient *= log_input
            derivative = coefficient * flat_result
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
            multipartials[1].append(contracted_tensor)

        shaped_partials: ShapedPartials = (multipartials[1], tuple(input.shape))
        multipartials[1] = unbroadcast(
            shaped_partials=shaped_partials, output_shape=tuple(result.shape)
        )[0]

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
