# python 3.12

# Standard Library dependencies
import math
import warnings
from itertools import permutations
from typing import Optional, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.utils.types import Partials, ShapedPartials


def start_partials(tensor: Tensor, order: int, device: torch.device) -> ShapedPartials:
    partials_list: list[Tensor] = []
    numel: int = tensor.numel()
    for i in range(order):
        if i == 0:
            partial: Tensor = torch.eye(numel, device=device)
        else:
            partial: Tensor = torch.zeros(
                size=tuple([numel for _ in range(i + 2)]), device=device
            )
        partials_list.append(partial)
    partials: Partials = tuple(partials_list)
    shape: Tuple[int, ...] = tuple(tensor.shape)
    shaped_partials: ShapedPartials = (partials, shape)
    return shaped_partials


def get_backward_idx(tensor: torch.Tensor) -> Optional[int]:
    dummy: Tensor = tensor.sum()
    idx: Union[None, int] = None
    # Check the position in next_functions where our tensor's grad_fn appears
    for grad_fn, pos in dummy.grad_fn.next_functions:
        if grad_fn is tensor.grad_fn:
            idx = pos
    dummy.detach()
    return idx


def sum_partials(partials_list: list[ShapedPartials]) -> ShapedPartials:
    """
    Sums a list of ShapedPartials by summing corresponding Partials tensors.

    Args:
        partials_list (List[ShapedPartials]): A list of ShapedPartials to be summed.

    Returns:
        ShapedPartials: The summed ShapedPartials.
    """
    if not partials_list:
        raise ValueError("The partials_list cannot be empty.")

    # Ensure all ShapedPartials have the same number of Partials
    num_partials_set: set[int] = {len(p[0]) for p in partials_list}
    if len(num_partials_set) != 1:
        raise ValueError("All ShapedPartials must have the same number of Partials.")

    # Ensure all ShapedPartials have the same shape
    shapes_set: set[Tuple[int, ...]] = {p[1] for p in partials_list}
    if len(shapes_set) != 1:
        raise ValueError("All ShapedPartials must have the same shape.")

    # Use zip to aggregate corresponding Partials and sum them
    aggregated_partials = zip(*(p[0] for p in partials_list))

    summed_partials = tuple(sum(tensor_group) for tensor_group in aggregated_partials)

    # Retrieve the common shape from the first element
    common_shape: Tuple[int] = partials_list[0][1]

    return (summed_partials, common_shape)


def unbroadcast(
    shaped_partials: ShapedPartials, output_shape: Tuple[int, ...]
) -> ShapedPartials:

    def is_broadcastable(shape: Tuple[int, ...], target: Tuple[int, ...]) -> bool:
        return all(math.gcd(s, t) == t for s, t in zip(shape, target))

    shape: Tuple[int, ...] = shaped_partials[1]
    padding: int = max((len(shape) - len(output_shape)), 0)
    target: Tuple[int, ...] = (1,) * padding + output_shape
    new_shaped_partials: ShapedPartials

    if shape == target:

        new_shaped_partials = shaped_partials

    elif is_broadcastable(shape=shape, target=target):

        # sumar las dimensiones que sean 1's en target
        new_shaped_partials = _unbroadcast_aux(
            shaped_partials=shaped_partials, output_shape=target
        )

    elif math.gcd(math.prod(shape), math.prod(target)) == math.prod(target):

        # Generate all broadcastable permutations
        perms = list(permutations(shape))
        broadcastable_perms: list[Tuple[int, ...]] = [
            p for p in perms if is_broadcastable(p, target)
        ]

        if len(broadcastable_perms) == 0:
            raise ValueError(
                "XAF found an intractable combination of permutation and broadcasting. "
                "Consider being more explicit in the arrangement of dimensions."
            )

        if len(broadcastable_perms) > 1:
            warnings.warn(
                "XAF found an ambiguous combination of permutation and broadcasting. "
                "This can lead to errors in partials computations. Consider being "
                "more explicit in the arrangement of dimensions.",
                RuntimeWarning,
            )

        # select the best attending to 2 criteria:
        # 1. The fewer swaps the better
        # 2. Swaps in the last dimensions are better
        def score(perm: Tuple[int, ...]) -> Tuple[int, int]:
            movement: int = sum(1 for p, a in zip(perm, target) if p != a)
            positions: list[int] = [
                i for i, (p, a) in enumerate(zip(perm, target)) if p != a
            ]
            return (movement, sum(positions))

        best_permutation: Tuple[int] = min(broadcastable_perms, key=score)
        new_shaped_partials = _unbroadcast_aux(
            shaped_partials=shaped_partials, output_shape=best_permutation
        )

    else:

        raise ValueError(
            "XAF found an intractable combination of permutation and broadcasting. "
            "Consider being more explicit in the arrangement of dimensions."
        )

    return new_shaped_partials


def _unbroadcast_aux(
    shaped_partials: ShapedPartials, output_shape: Tuple[int, ...]
) -> ShapedPartials:

    shape: Tuple[int, ...] = shaped_partials[1]
    target: Tuple[int, ...] = output_shape
    list_new_partials: list[Tensor] = list()
    new_shape: Tuple[int, ...] = tuple([s for s in output_shape if not s == 1])

    for partial in shaped_partials[0]:

        view: list[int] = [partial.shape[0]]
        new_view: list[int] = [partial.shape[0]]
        broadcasts: list[bool] = [False]
        for _ in partial.shape[1:]:
            view.extend(shape)
            new_view.append(math.prod(new_shape))
            broadcasts.extend([not s == t for s, t in zip(shape, target)])

        if len(new_shape) == 0:
            new_view = (partial.shape[0], 1)
            new_shape = (1,)

        dims: list[bool] = [i for i, broadcast in enumerate(broadcasts) if broadcast]
        reshaped_partial: Tensor = partial.view(size=tuple(view))
        reshaped_partial = reshaped_partial.sum(dim=tuple(dims))
        reshaped_partial = reshaped_partial.view(new_view)
        list_new_partials.append(reshaped_partial)

    new_shaped_partials: ShapedPartials = (tuple(list_new_partials), new_shape)

    assert len(new_shape) > 0, partial.ndim

    return new_shaped_partials


def unscale(shaped_partials: ShapedPartials, target_numel: int) -> ShapedPartials:
    """
    Variant of unbroadcast_partials that *requires* the final shape is exactly
    (batch_size, target_numel). If no exact subshape can produce 'target_numel'
    elements, raise an error.
    """
    import math

    print([T.shape for T in shaped_partials[0]])
    print(shaped_partials[1], target_numel)

    shape: Tuple[int, ...] = shaped_partials[1]

    # 1) Find subshape ...
    target_shape: Tuple[int, ...] = _find_closest_subshape(shape, target_numel)
    print("Selected subshape:", target_shape)
    found_numel: int = math.prod(target_shape)

    if found_numel != target_numel:
        raise ValueError(
            f"Cannot find any subshape of {shape} whose product is exactly "
            f"{target_numel}. Closest subshape is {target_shape} with "
            f"product={found_numel}."
        )

    list_new_partials: list[Tensor] = []
    for idx, partial in enumerate(shaped_partials[0]):
        batch_size: int = partial.shape[0]

        # A) Reshape to (batch_size, *shape)
        reshaped_partial = partial.view(batch_size, *((idx + 1) * shape))

        # B) Identify dims to reduce
        broadcast_dims: list[int] = [
            i + 1
            for i, (s_dim, t_dim) in enumerate(zip(shape, target_shape))
            if (t_dim == 1 and s_dim != 1)
        ]

        # C) Sum/prod
        if len(broadcast_dims) > 0:
            reshaped_partial = reshaped_partial.sum(dim=broadcast_dims)

        # D) Reshape to (batch_size, target_numel)
        final_view: Tuple[int, ...] = (batch_size, *((idx + 1) * (target_numel,)))
        reshaped_partial: Tensor = reshaped_partial.view(final_view)
        list_new_partials.append(reshaped_partial)

    new_shaped_partials: ShapedPartials = (tuple(list_new_partials), (target_numel,))
    return new_shaped_partials


def _find_closest_subshape(
    shape: Tuple[int, ...], target_numel: int
) -> Tuple[int, ...]:
    """
    From `shape`, generate all subshapes obtained by setting some dims to 1.
    Return the one whose product is >= target_numel and closest to it.
    If none is >=, return the largest possible (which is the original shape).
    Subshapes that set left dims to 1 are explored first.
    """
    best_subshape: Tuple[int] = shape
    best_product: int = math.prod(shape)

    def backtrack(i: int, current_shape: list[int]) -> None:
        nonlocal best_subshape, best_product

        if i == len(shape):
            current_prod: int = math.prod(current_shape)
            # We want >= target_numel but as close as possible above it
            if current_prod >= target_numel:
                if current_prod < best_product:
                    best_product = current_prod
                    best_subshape = tuple(current_shape)
            return

        # 1) Try setting this dimension to 1
        original_dim: int = current_shape[i]
        current_shape[i] = 1
        backtrack(i + 1, current_shape)

        # 2) Restore and try the original size
        current_shape[i] = shape[i]
        backtrack(i + 1, current_shape)

        current_shape[i] = original_dim

    # Start recursive search
    shape_list = list(shape)
    backtrack(0, shape_list)

    return best_subshape
