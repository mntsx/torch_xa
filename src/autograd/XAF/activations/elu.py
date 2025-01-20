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


class EluXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[float, int, int, Tensor]:
        saved_alpha: float = self.grad_fn._saved_alpha
        saved_input_scale: int = self.grad_fn._saved_input_scale
        saved_scale: int = self.grad_fn._saved_scale
        saved_self: Tensor = self.grad_fn._saved_self.to(device=self._device)
        return (saved_alpha, saved_input_scale, saved_scale, saved_self)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0
        ctx: Tuple[Tensor, Tensor] = self._get_context()
        alpha: Tensor = ctx[0]
        input: Tensor = ctx[3]

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

        # obtain element wise internal first derivative tensor
        flat_input: Tensor = input.flatten()
        cond: Tensor = flat_input > 0
        t1: Tensor = torch.tensor([1.0], device=self._device)
        t0: Tensor = alpha * torch.exp(flat_input)
        derivative: Tensor = torch.where(condition=cond, input=t1, other=t0)
        derivatives: list[Tensor] = [derivative for _ in range(self._order)]

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
