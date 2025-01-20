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


class ProdXBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tuple[int, ...], bool, Tensor, Tensor]:
        saved_result: Tensor = self.grad_fn._saved_result.to(device=self._device)
        saved_self: Tensor = self.grad_fn._saved_self.to(device=self._device)
        saved_dim: Tuple[int, ...] = tuple([i for i, _ in enumerate(saved_self.shape)])
        saved_keepdim: bool = False
        return (saved_dim, saved_keepdim, saved_result, saved_self)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        ctx: Tuple[Tuple[int, ...], bool, Tensor, Tensor]
        ctx = self._get_context()
        dim: Tuple[int, ...] = ctx[0]
        keepdim: bool = ctx[1]
        result: Tensor = ctx[2]
        input: Tensor = ctx[3]

        batch_dims: list[int] = list()
        expected_output_shape: list[int] = list()
        for i, size in enumerate(input.shape):
            if i in dim:
                if keepdim:
                    expected_output_shape.append(1)
            else:
                batch_dims.append(i)
                expected_output_shape.append(size)

        if len(expected_output_shape) == 0:
            expected_output_shape = [1]

        print(len(shaped_output_partials[0]))

        shaped_output_partials = unbroadcast(
            shaped_partials=shaped_output_partials,
            output_shape=tuple(expected_output_shape),
        )
        output_partials: Partials = shaped_output_partials[0]
        output_shape: Tuple[int, ...] = shaped_output_partials[1]
        assert len(output_partials) == self._order

        shape: Tuple[int, ...]
        permutation: list[int]
        graph_output_numel: int = output_partials[0].shape[0]
        multipartials: list[list[Tensor]] = [[]]
        multishapes: list[Tuple[int, ...]] = [input.shape]
        expressions: list[SumGroup]
        expressions = [calculate_n_order_partial(n=n + 1) for n in range(self._order)]

        # compute input internal partials
        derivatives: list[Tensor] = list()
        batch_size: int = math.prod([input.shape[i] for i in batch_dims])
        prod_size: int = math.prod([input.shape[i] for i in dim])
        for order in range(1, self._order + 1):
            derivative: Tensor
            if order == 1:
                shape = tuple([1 if d in dim else s for d, s in enumerate(input.shape)])
                unsqueezed_result: Tensor = result.view(size=shape)
                expanded_result: Tensor = unsqueezed_result.expand_as(other=input)
                quotient: Tensor = expanded_result / input
                permutation = [*batch_dims, *dim]
                permuted_quotient: Tensor = quotient.permute(dims=tuple(permutation))
                derivative = permuted_quotient.view(size=(batch_size, 1, prod_size))
            else:
                shape = (batch_size, 1, *(order * (prod_size,)))
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
                batch=(True, True),
            )

            # obtain reshape list
            reshape_list: list[int] = [graph_output_numel]
            for _ in range(i + 1):
                reshape_list.extend([sz for sz in output_shape if not sz == 1])
                reshape_list.extend([input.shape[d] for d in dim])

            # obtain permutation
            dims: list[int] = [d for d, _ in enumerate(input.shape) if d not in dim]
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
                unshape_list.append(math.prod(input.shape))

            reshaped_tensor: Tensor = contracted_tensor.view(size=tuple(reshape_list))
            permuted_tensor: Tensor = reshaped_tensor.permute(dims=tuple(perm_list))
            unshaped_tensor: Tensor = permuted_tensor.reshape(shape=tuple(unshape_list))
            multipartials[0].append(unshaped_tensor)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None


class ProdXBackward1(ProdXBackward0):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def _get_context(self) -> Tuple[Tuple[int, ...], bool, Tensor, Tensor]:
        saved_dim: Tuple[int, ...] = (self.grad_fn._saved_dim,)
        saved_keepdim: bool = self.grad_fn._saved_keepdim
        saved_result: Tensor = self.grad_fn._saved_result.to(device=self._device)
        saved_self: Tensor = self.grad_fn._saved_self.to(device=self._device)
        return (saved_dim, saved_keepdim, saved_result, saved_self)
