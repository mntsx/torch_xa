# python 3.12

# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop import contraction
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unbroadcast
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class DotXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor, Tensor]:
        v1: Tensor = self._grad_fn._saved_self.to(device=self._device)
        v2: Tensor = self._grad_fn._saved_tensor.to(device=self._device)
        return (v1, v2)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0

        ctx: Tuple[Tensor, Tensor] = self._get_context()
        v1: Tensor = ctx[0]
        v2: Tensor = ctx[1]

        expected_output_shape: Tuple[int, ...] = (1,)
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials, output_shape=expected_output_shape
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order
        assert len(output_shape) == 1
        assert output_shape[0] == 1

        multipartials: list[list[Tensor]] = [[], []]
        multishapes: list[Tuple[int, ...]] = [(v1.shape[0],), (v2.shape[0],)]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # retrieve some data
        internal_partial: Tensor
        internal_partials: list[Tensor]

        # compute m1 internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            if order == 1:
                internal_partial = v2.unsqueeze(0)
            else:
                internal_partial = torch.zeros(size=(1,), device=self._device)
            internal_partials.append(internal_partial)

        # compute m1 partials
        pretensors: Tuple[Tensor, ...]
        subtensors: Tuple[Tensor, ...]
        contracted_tensor: Tensor
        pretensors = output_partials
        subtensors = tuple(internal_partials)

        for i, expression in enumerate(expressions):
            contracted_tensor: Tensor = contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
                batch=(False, False),
            )
            multipartials[0].append(contracted_tensor)

        # compute m2 internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            if order == 1:
                internal_partial = v1.unsqueeze(0)
            else:
                internal_partial = torch.zeros(size=(1,), device=self._device)
            internal_partials.append(internal_partial)

        # compute m2 partials
        pretensors = output_partials
        subtensors = tuple(internal_partials)
        for _, expression in enumerate(expressions):
            contracted_tensor: Tensor = contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
                batch=(False, False),
            )
            multipartials[1].append(contracted_tensor)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
