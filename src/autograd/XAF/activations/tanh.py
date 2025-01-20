# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop import diagonal_contraction
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.engine.symbolic.polinomial import (
    poly_derivative,
    poly_eval,
    poly_var_mul,
)
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unbroadcast
from src.utils.types import AutogradFunction, ShapedPartials, Partials


_tanh_poly_cache: dict[int, list[float]] = {}  # store T_n(t) as a list of coefficients
# Define T_1(t) = 1 - t^2 => [1.0, 0.0, -1.0] (i.e. 1 + 0*t - 1*t^2)
_tanh_poly_cache[1] = [1.0, 0.0, -1.0]


def get_tanh_poly(n: int) -> list[float]:
    """
    Returns the list of coefficients of T_n(t) where
    T_n(t) = d^n/dx^n [tanh(x)], represented in t = tanh(x).
    """
    if n in _tanh_poly_cache:
        return _tanh_poly_cache[n]

    max_cached: int = max(_tanh_poly_cache.keys())
    for k in range(max_cached, n):
        Tk: list[float] = _tanh_poly_cache[k]
        dTk: list[float] = poly_derivative(Tk)  # d/dt [T_k(t)]

        # (1 - t^2) * dTk
        # (1 - t^2) => [1.0, 0.0, -1.0]
        T_next: list[float] = poly_var_mul(dTk, [1.0, 0.0, -1.0])
        _tanh_poly_cache[k + 1] = T_next

    return _tanh_poly_cache[n]


def tanh_derivate(tensor: Tensor, n: int) -> Tensor:
    """
    Returns the n-th derivative of tanh(x) evaluated at inv_tanh(tensor).
    """
    if n == 0:
        # "0-th derivative" => tanh(x) itself
        return tensor

    # Obtain T_n(t)
    Tn: list[float] = get_tanh_poly(n)
    # Evaluate at t
    return poly_eval(Tn, tensor)


class TanhXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor]:
        saved_result: Tensor = self._grad_fn._saved_result
        return (saved_result,)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        ctx: Tuple[Tensor, Tensor] = self._get_context()
        result: Tensor = ctx[0]

        expected_output_shape: Tuple[int, ...] = tuple(result.shape)
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials,
            output_shape=expected_output_shape,
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        multipartials: list[list[Tensor]] = [[]]
        multishapes: list[Tuple[int, ...]] = [output_shape]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=(n + 1)) for n in range(self._order)]

        # compute internal partials
        derivatives: list[Tensor] = list()
        for order in range(1, self._order + 1):
            derivative: Tensor = tanh_derivate(tensor=result.flatten(), n=order)
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
