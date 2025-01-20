# python 3.12

# Standard Library dependencies
import math
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


def log_derivate(tensor: Tensor, n: int) -> Tensor:
    """
    Returns the n-th derivative of log(x) evaluated at x = `tensor`.

    Formula:
      d^n/dx^n [log(x)] =
         log(x),                       n = 0
         (-1)^(n-1) * (n-1)! / x^n,    n >= 1
    """
    if n == 0:
        return torch.log(tensor)  # 0th derivative is log(x) itself
    else:
        # Use the closed-form for n >= 1
        sign: int = (-1) ** (n - 1)
        factorial_term: int = math.factorial(n - 1)
        return sign * factorial_term / (tensor**n)


class LogXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor]:
        saved_self: Tensor = self._grad_fn._saved_self.to(device=self._device)
        return (saved_self,)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        input: Tensor = self._get_context()[0]

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

        # obtain element wise internal first derivative tensor
        derivatives: list[Tensor] = list()
        for order in range(1, self._order + 1):
            derivatives.append(log_derivate(tensor=flat_input, n=order))

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
