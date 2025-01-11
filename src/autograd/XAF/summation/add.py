# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop import hadamard
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class AddXBackward0(ExtendedAutogradFunction):

    def __init__(self, grad_fn: AutogradFunction, order: int) -> None:
        super().__init__(grad_fn=grad_fn, order=order)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[int, ...]:
        saved_alpha: float = self.grad_fn._saved_alpha
        return (saved_alpha,)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        alpha: float = self._get_context()[0]

        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        derivative: Tensor
        derivatives: list[Tensor]
        output_numel: int = shaped_output_partials[0][0].shape[1]
        multipartials: list[list[Tensor]] = [[], []]
        multishapes: list[Tuple[int, ...]] = [output_shape, output_shape]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # compute other internal partials
        derivatives = list()
        for order in range(1, self._order + 1):
            if order == 1:
                derivative = torch.ones(size=(output_numel,))
            else:
                derivative = torch.zeros(size=(output_numel,))
            derivatives.append(derivative)

        # compute other partials
        pretensors: Tuple[Tensor, ...]
        subtensors: Tuple[Tensor, ...]
        pretensors = output_partials
        subtensors = tuple(derivatives)
        for expression in expressions:
            contracted_tensor: Tensor = hadamard(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
            )
            multipartials[0].append(contracted_tensor)

        # compute other internal partials
        derivatives = list()
        for order in range(1, self._order + 1):
            if order == 1:
                derivative = alpha * torch.ones(size=(output_numel,))
            else:
                derivative = torch.zeros(size=(output_numel,))
            derivatives.append(derivative)

        # compute other partials
        pretensors = output_partials
        subtensors = tuple(derivatives)
        for expression in expressions:
            contracted_tensor = hadamard(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
            )
            multipartials[1].append(contracted_tensor)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
