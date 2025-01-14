# python 3.12

# Standard Library dependencies
import math
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop import contractor
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction, ShapedPartials, Partials


def softmax_derivate(input: Tensor, n: int, device: torch.device) -> Tensor:
    """
    Computes the n-th derivative of the softmax output with respect to its inputs
    in a fully tensorized manner, returning a tensor of shape:
        [batch_size, C, C, ..., C]  (with n copies of "C" at the end).

    Arguments:
    ----------
      input: Tensor of shape [B, C].
             Assumed to be softmax probabilities, i.e., s[b, c].
      n    : Order of the derivative.
             - If n=0, returns `input` itself (shape [B, C]).
             - If n>0, returns a tensor of shape [B, C, C, ..., C] (n copies of C).

    Returns:
    --------
      out[b, i, j1, ..., jn] = (∂^n / ∂x_{j1} ... ∂x_{jn}) s_i(b).

    Notes:
    ------
      - For large n or large C, memory usage grows as O(B * C^(n+1)).
      - We remove explicit Python loops over output dimension by leveraging broadcasting
    """
    s: Tensor = input  # [B, C]
    B: int
    C: int
    B, C = s.shape

    # If n=0, the "0th derivative" is just s
    if n == 0:
        return s

    # ----------------------------------------------------
    # 1) Meshgrid enumerating (j1, j2, ..., jn) in {0..C-1}^n
    #    => mesh_idx: shape [n, C, C, ..., C] (n copies of C)
    # ----------------------------------------------------
    mesh: Tuple[Tensor] = torch.meshgrid(
        *[torch.arange(C, device=device) for _ in range(n)], indexing="ij"
    )
    mesh_idx: Tensor = torch.stack(mesh, dim=0)  # shape => [n, C, C, ..., C]

    # = (C, C, ..., C)  (n dims)
    partial_shape: Tuple[int, ...] = tuple(mesh_idx.shape[1:])

    # ----------------------------------------------------
    # 2) Build product of s_j1 * s_j2 * ... * s_jn, fully batched
    # ----------------------------------------------------
    # Flatten each dimension so that mesh_idx has shape [n, C^n] when flattened.
    mesh_stacked: Tensor = mesh_idx.view(n, -1)  # [n, C^n]
    prod_s_shape: Tuple[int, ...] = (B,) + partial_shape  # e.g. (B, C, C, ..., C)

    # Expand to [B, n, C^n] so we can gather from s => shape [B, n, C^n].
    mesh_stacked_expanded: Tensor = mesh_stacked.unsqueeze(0).expand(B, -1, -1)

    # Expand s to shape [B, n, C] => gather => [B, n, C^n]
    s_expanded: Tensor = s.unsqueeze(1).expand(-1, n, -1)
    prod_all_flat: Tensor = s_expanded.gather(dim=2, index=mesh_stacked_expanded)

    # Multiply along "n" => [B, C^n], then reshape => [B, *partial_shape]
    prod_s_flat: Tensor = prod_all_flat.prod(dim=1)
    prod_s_jmulti: Tensor = prod_s_flat.view((B,) + partial_shape)  # => [B, C, ..., C]

    # ----------------------------------------------------
    # 3) Count how many times each class i appears in (j1..jn)
    #    => counts: shape [C, C, ..., C]
    # ----------------------------------------------------
    counts: Tensor = torch.zeros((C,) + partial_shape, dtype=torch.long, device=device)
    # Use scatter_add_ on dim=0 => "class" dimension
    counts.scatter_add_(
        dim=0,
        index=mesh_idx,
        src=torch.ones_like(mesh_idx, dtype=torch.long, device=device),
    )
    # counts[i, j1, ..., jn] = number of times i appears in (j1..jn).

    # ----------------------------------------------------
    # 4) Build the final derivative:
    #      d^n s_i / (dx_{j1}..dx_{jn})
    #        = comb(n, count_i) * s_i^(1 - count_i) * (-1)^(n - count_i) * (∏ s_{ j_k })
    # ----------------------------------------------------
    comb_table: list[int] = [math.comb(n, r) for r in range(n + 1)]
    sign_table: list[int] = [(-1) ** (n - r) for r in range(n + 1)]

    comb_t: Tensor = torch.tensor(comb_table, device=device, dtype=s.dtype)
    sign_t: Tensor = torch.tensor(sign_table, device=device, dtype=s.dtype)

    # The output needs shape => (B, C) + partial_shape, i.e. (n+2) dims total.
    out_shape: Tuple[int, ...] = (B, C) + partial_shape
    out: Tensor = torch.zeros(out_shape, dtype=s.dtype, device=device)

    # Loop over each possible "output class" i
    for i in range(C):
        # counts[i] => shape partial_shape => (C, ..., C)
        count_i: Tensor = counts[i]  # how many times i appears in (j1..jn)

        # binomial factor + sign => shape (C, ..., C)
        binom_factor: Tensor = comb_t[count_i]  # index into comb_table
        sign_factor: Tensor = sign_t[count_i]  # index into sign_table

        # s[:, i] => shape [B], expand to [B, 1, 1, ..., 1]
        s_i_scalar: Tensor = s[:, i].view(B, *([1] * n))  # (B,) + (1,)*n
        exponent: Tensor = 1 - count_i  # shape => partial_shape, integer

        # Expand to match prod_s_shape => (B,) + partial_shape
        s_i_expanded: Tensor = s_i_scalar.expand(prod_s_shape)

        # Elementwise power
        s_i_pow: Tensor = s_i_expanded**exponent

        # Multiply everything => shape => (B,) + partial_shape
        # val: Tensor = binom_factor * sign_factor * s_i_pow * prod_s_jmulti
        val: Tensor = binom_factor
        val *= sign_factor
        val = val * s_i_pow
        val *= prod_s_jmulti

        # out[:, i] => shape => (B,) + partial_shape
        out[:, i] = val

    return out


class SoftmaxBackward0(ExtendedAutogradFunction):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def integral(self) -> bool:
        integral: bool = 0 in self._output_registry
        return integral

    def _get_context(self) -> Tuple[Tuple[int, ...], Tensor]:
        saved_dim: Tuple[int, ...] = self._grad_fn._saved_dim
        saved_result: Tensor = self._grad_fn._saved_result
        return (saved_dim, saved_result)

    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        assert idx == 0
        ctx: Tuple[Tensor, Tensor] = self._get_context()
        dim: int = ctx[0]
        result: Tensor = ctx[1]

        expected_output_shape: Tuple[int, ...] = tuple(result.shape)
        shaped_output_partials = self._unbroadcast_partials(
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

        # retrieve some data
        internal_partial: Tensor
        internal_partials: list[Tensor]
        graph_output_numel: int = output_partials[0].shape[0]

        # define permutations data
        dim_size: int = result.shape[dim]
        batch_size: int = int(result.numel() / dim_size)
        ndim: int = result.ndim
        permutation: Tuple[int, ...] = (*[d for d in range(ndim) if not d == dim], dim)
        # permutation2: Tuple[int, ...] = (*range(dim), ndim - 1, *range(dim, ndim - 1))

        # compute internal partials
        internal_partials = list()
        for order in range(1, self._order + 1):
            permuted_result: Tensor = result.permute(permutation)
            reshaped_result: Tensor = permuted_result.view(batch_size, dim_size)
            internal_partial = softmax_derivate(
                input=reshaped_result, n=order, device=self._device
            )
            internal_partials.append(internal_partial)

        # compute partials
        aux: list[Tensor] = list()
        permuted_shapes: list[Tuple[int, ...]] = list()
        for i, partial in enumerate(output_partials):

            # obtain shape list
            reshape_list: list[int] = [graph_output_numel]
            for _ in range(i + 1):
                reshape_list.extend(result.shape)
            # obtain permutation list
            pointer = 1
            perm_list: list[int] = [0]
            for _ in range(i + 1):
                perm_list.extend([p + pointer for p in permutation])
                pointer += ndim
            # obtain unshape list
            unshape_list: list[int] = [graph_output_numel]
            for _ in range(i + 1):
                unshape_list.extend([batch_size, dim_size])

            # reshape partial to: [Graph_output_shape, (i+1)*(batch, contracting_dim)]
            # * contracting_dim of partial is partial.shape[dim]
            reshaped_partial: Tensor = partial.view(size=tuple(reshape_list))
            permuted_partial: Tensor = reshaped_partial.permute(dims=tuple(perm_list))
            permuted_shapes.append(tuple(permuted_partial.shape))
            unshape: Tuple[int, ...] = tuple(unshape_list)
            unshaped_partial: Tensor = permuted_partial.reshape(shape=unshape)
            aux.append(unshaped_partial)

        pretensors = tuple(aux)
        subtensors = tuple(internal_partials)
        for i, expression in enumerate(expressions):
            contracted_tensor: Tensor = contractor(
                pretensors=pretensors,
                subtensors=subtensors,
                expression=expression,
                batch=(True, True),
                device=self._device,
            )
            # recover output_partials shape (rearranging dimensions properly)
            reshaped_tensor: Tensor = contracted_tensor.view(permuted_shapes[i])
            pointer = 1
            perm_list: list[int] = [0]
            for _ in range(i + 1):
                perm_list.extend([p + pointer for p in permutation])
                pointer += ndim
            permuted_tensor: Tensor = reshaped_tensor.permute(dims=tuple(perm_list))
            shape: Tuple[int, ...] = tuple(output_partials[i].shape)
            reshaped_tensor: Tensor = permuted_tensor.reshape(shape=shape)
            reshaped_tensor = reshaped_tensor.contiguous()
            multipartials[0].append(reshaped_tensor)

        self._update_multipartials(multipartials=multipartials, shapes=multishapes)

        return None
