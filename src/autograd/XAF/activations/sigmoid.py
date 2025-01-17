# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop import hadamard
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.engine.symbolic.polinomial import (
    poly_add,
    poly_derivative,
    poly_eval,
    poly_var_mul,
)
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unbroadcast
from src.utils.types import AutogradFunction, ShapedPartials, Partials


# Polynomial cache for Q_n(s). Q_n(s) is stored as a list of coefficients.
_sigmoid_poly_cache: dict[int, list[float]] = {1: [1.0]}
# Q_1(s) = 1


def get_sigmoid_poly(n: int) -> list[float]:
    """
    Returns the list of coefficients of Q_n(s) such that
    sigma^{(n)}(x) = s(1-s)*Q_n(s).
    """
    if n in _sigmoid_poly_cache:
        return _sigmoid_poly_cache[n]

    # We build recursively from what we already have
    max_cached: int = max(_sigmoid_poly_cache.keys())
    for k in range(max_cached, n):
        Qk: list[float] = _sigmoid_poly_cache[k]  # Q_k
        dQk: list[float] = poly_derivative(Qk)  # Q_k'(s)

        # (1 - 2s)*Q_k(s)
        # polynomial (1) - 2*s => [1.0, -2.0]
        part1: list[float] = poly_var_mul(Qk, [1.0, -2.0])

        # s(1-s)*Q_k'(s)
        # s(1-s) => polynomial: [0.0, 1.0, -1.0]
        part_s1s: list[float] = [0.0, 1.0, -1.0]
        part2: list[float] = poly_var_mul(dQk, part_s1s)

        # Q_{k+1}(s) = part1 + part2
        Q_next: list[float] = poly_add(part1, part2)
        _sigmoid_poly_cache[k + 1] = Q_next

    return _sigmoid_poly_cache[n]


def sigmoid_derivate(tensor: Tensor, n: int) -> Tensor:
    """
    Returns the n-th derivative of sigma(x) evaluated at inv_sigmoid(tensor).
    All vectorized in PyTorch.
    """

    if n == 0:
        # For consistency, the "0-th derivative" is the function itself
        return tensor

    # Q_n(s) in the form of coefficients
    Qn: list[float] = get_sigmoid_poly(n)

    # Evaluate Q_n(s) at s
    poly_val: Tensor = poly_eval(Qn, tensor)

    # Multiply by s*(1-s)
    return tensor * (1 - tensor) * poly_val


class SigmoidXBackward0(ExtendedAutogradFunction):

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
            derivative: Tensor = sigmoid_derivate(tensor=result.flatten(), n=order)
            derivatives.append(derivative)

        # compute partials
        pretensors: Tuple[Tensor, ...] = output_partials
        subtensors: Tuple[Tensor, ...] = tuple(derivatives)
        for expression in expressions:
            contracted_tensor: Tensor = hadamard(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
            )
            multipartials[0].append(contracted_tensor)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
