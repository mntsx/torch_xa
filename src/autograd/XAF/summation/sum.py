# python 3.12

# Standard Library dependencies
import math
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.engine.backprop import contraction
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import unbroadcast
from src.utils.types import AutogradFunction, ShapedPartials, Partials


class SumXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tuple[int, ...], bool, Tuple[int, ...]]:
        saved_self_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_self_sym_sizes
        saved_dim: Tuple[int, ...] = tuple(
            [i for i, _ in enumerate(saved_self_sym_sizes)]
        )
        saved_keepdim: bool = False
        return (saved_dim, saved_keepdim, saved_self_sym_sizes)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        ctx: Tuple[Tuple[int, ...], bool, Tuple[int, ...]]
        ctx = self._get_context()
        dim: Tuple[int, ...] = ctx[0]
        keepdim: bool = ctx[1]
        sym_sizes: Tuple[int, ...] = ctx[2]

        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        batch_dims: list[int] = list()
        expected_output_shape: list[int] = list()
        for i, size in enumerate(sym_sizes):
            if i in dim:
                if keepdim:
                    expected_output_shape.append(1)
            else:
                batch_dims.append(i)
                expected_output_shape.append(size)

        if len(expected_output_shape) == 0:
            expected_output_shape = [1]

        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials,
            output_shape=tuple(expected_output_shape),
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        shape: Tuple[int, ...]
        graph_output_numel: int = output_partials[0].shape[0]
        multipartials: list[list[Tensor]] = [[]]
        multishapes: list[Tuple[int, ...]] = [sym_sizes]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # compute input internal partials
        derivatives: list[Tensor] = list()
        sum_size: int = math.prod([sym_sizes[i] for i in dim])
        for order in range(1, self._order + 1):
            derivative: Tensor
            shape = (1, *(order * (sum_size,)))
            if order == 1:
                derivative = torch.ones(size=shape, device=self._device)
            else:
                derivative = torch.zeros(size=shape, device=self._device)
            derivatives.append(derivative)

        # compute input partials
        pretensors: Tuple[Tensor, ...]
        subtensors: Tuple[Tensor, ...]
        batch_size: int = math.prod(expected_output_shape)
        batched_partials: list[Tensor] = list()
        for i, partial in enumerate(output_partials):
            shape = (graph_output_numel, *((i + 1) * (batch_size, 1)))
            batched_partial: Tensor = partial.view(size=shape)
            batched_partials.append(batched_partial)
        pretensors = tuple(batched_partials)
        subtensors = tuple(derivatives)
        for i, expression in enumerate(expressions):
            contracted_tensor: Tensor = contraction(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                device=self._device,
                batch=(True, False),
            )

            # obtain reshape list
            reshape_list: list[int] = [graph_output_numel]
            for _ in range(i + 1):
                reshape_list.extend([sz for sz in output_shape if not sz == 1])
                reshape_list.extend([sym_sizes[d] for d in dim])

            # obtain permutation
            dims: list[int] = [d for d, _ in enumerate(sym_sizes) if d not in dim]
            dims.extend(dim)
            permutation: list[int] = [0 for _ in dims]
            for index, value in enumerate(dims):
                permutation[value] = index

            # obtain permutaion list
            pointer: int = 1
            perm_list: list[int] = [0]
            for _ in range(i + 1):
                perm_list.extend([p + pointer for p in permutation])
                pointer += len(permutation)

            # obtain unshape list
            unshape_list: list[int] = [graph_output_numel]
            for _ in range(i + 1):
                unshape_list.append(math.prod(sym_sizes))

            reshaped_tensor: Tensor = contracted_tensor.view(size=tuple(reshape_list))
            permuted_tensor: Tensor = reshaped_tensor.permute(dims=tuple(perm_list))
            unshaped_tensor: Tensor = permuted_tensor.reshape(shape=tuple(unshape_list))
            multipartials[0].append(unshaped_tensor)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None


class SumXBackward1(SumXBackward0):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def _get_context(self) -> Tuple[Tuple[int, ...], bool, Tuple[int, ...]]:
        saved_dim: Tuple[int, ...] = self._grad_fn._saved_dim
        saved_keepdim: bool = self._grad_fn._saved_keepdim
        saved_self_sym_sizes: Tuple[int, ...] = self._grad_fn._saved_self_sym_sizes
        return (saved_dim, saved_keepdim, saved_self_sym_sizes)
