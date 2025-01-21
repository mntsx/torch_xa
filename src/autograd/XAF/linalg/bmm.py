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


class BmmXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tensor, Tensor]:
        m1: Tensor = self._grad_fn._saved_self.to(device=self._device)
        m2: Tensor = self._grad_fn._saved_mat2.to(device=self._device)
        return (m1, m2)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:

        assert idx == 0

        ctx: Tuple[Tensor, Tensor] = self._get_context()
        m1: Tensor = ctx[0]
        m2: Tensor = ctx[1]

        expected_output_shape: Tuple[int, ...] = (m1.shape[0], m1.shape[1], m2.shape[2])
        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials, output_shape=expected_output_shape
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_shape) == 3
        assert len(output_partials) == self._order

        multipartials: list[list[Tensor]] = [[], []]
        multishapes: list[Tuple[int, ...]] = [tuple(m1.shape), tuple(m2.shape)]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # precalculations
        internal_partial: Tensor
        internal_partials: list[Tensor]
        ones: Tensor
        shape: Tuple[int, ...]
        graph_output_numel: int = output_partials[0].shape[0]
        batch_size: int
        input_size: int

        # compute m1 internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            if order == 1:
                ones = torch.ones(size=(m1.shape[1],), device=self._device)
                repeated_m2: Tensor
                repeated_m2 = torch.einsum(m2, (0, 1, 2), ones, (3,), (0, 3, 2, 1))
                internal_partial = repeated_m2.flatten(start_dim=0, end_dim=1)
            else:
                internal_partial = torch.zeros(size=(1,), device=self._device)
            internal_partials.append(internal_partial)

        # compute m1 partials
        pretensors: Tuple[Tensor, ...]
        subtensors: Tuple[Tensor, ...]
        contracted_tensor: Tensor
        aux: list[Tensor] = list()
        for i, partial in enumerate(output_partials):
            batch_size = output_shape[0] * output_shape[1]
            input_size = output_shape[2]
            shape = (graph_output_numel, *((batch_size, input_size) * (i + 1)))
            aux.append(partial.reshape(shape=shape))
        pretensors = tuple(aux)
        subtensors = tuple(internal_partials)
        for i, expression in enumerate(expressions):
            contracted_tensor: Tensor = contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
                batch=(True, True),
            )
            dual_numel: int = m1.shape[0] * m1.shape[1] * m2.shape[1]
            shape = (graph_output_numel, *[dual_numel for _ in range(i + 1)])
            multipartials[0].append(contracted_tensor.reshape(shape=shape))

        # compute m2 internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            if order == 1:
                ones = torch.ones(size=(m2.shape[2],), device=self._device)
                repeated_m1: Tensor
                repeated_m1 = torch.einsum(m1, (0, 1, 2), ones, (3,), (0, 3, 1, 2))
                internal_partial = repeated_m1.flatten(start_dim=0, end_dim=1)
            else:
                internal_partial = torch.zeros(size=(1,), device=self._device)
            internal_partials.append(internal_partial)

        # compute m2 partials
        aux = list()
        for i, partial in enumerate(output_partials):
            # shape output partial to (batch, input, non-contracting dim)
            shape = (graph_output_numel, *(output_shape * (i + 1)))
            viewed_partial: Tensor = partial.view(size=shape)
            # permute 2nd and 3rd dimensions
            dims: list[int] = [0]
            pointer: int = 1
            for i in range(i + 1):
                dims.extend([pointer, pointer + 2, pointer + 1])
                pointer += 3
            permuted_partial: Tensor = viewed_partial.permute(dims=tuple(dims))
            # shape to (batch, input)
            batch_size = output_shape[0] * output_shape[2]
            input_size = output_shape[1]
            shape = (graph_output_numel, *((batch_size, input_size) * (i + 1)))
            aux.append(permuted_partial.reshape(shape=shape))
        pretensors = tuple(aux)
        subtensors = tuple(internal_partials)
        for i, expression in enumerate(expressions):
            contracted_tensor: Tensor = contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
                batch=(True, True),
            )
            # shape contracted tensor
            shape = (
                graph_output_numel,
                *((i + 1) * (m2.shape[0], m2.shape[2], m2.shape[1])),
            )
            viewed_tensor: Tensor = contracted_tensor.view(size=shape)
            # permute 2nd and 3rd dimensions
            dims: list[int] = [0]
            pointer: int = 1
            for i in range(i + 1):
                dims.extend([pointer, pointer + 2, pointer + 1])
                pointer += 3
            permuted_tensor: Tensor = viewed_tensor.permute(dims=dims)
            # reshape as partial
            dual_numel: int = m1.shape[0] * m1.shape[2] * m2.shape[2]
            shape = (graph_output_numel, *[dual_numel for _ in range(i + 1)])
            multipartials[1].append(permuted_tensor.reshape(shape=shape))

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
