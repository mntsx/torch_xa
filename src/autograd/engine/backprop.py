# python 3.12

# Standard Library dependencies
from typing import Optional, Tuple, Union

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from src.autograd.engine.symbolic.derivation import Partial, SumGroup
from src.tensors.functional import construct_nd_identity, einsum


def contractor(
    pretensors: Tuple[Tensor, ...],
    subtensors: Tuple[Tensor, ...],
    expression: SumGroup,
    batch: Optional[bool] = False,
) -> Tensor:

    expression.ensure_sorted()
    accum: Optional[Tensor] = None

    for product in expression.products:
        product.ensure_sorted()
        partials: list[Partial] = product.partials

        # Collect arguments for the einsum call
        einsum_args: list[Union[list[int], Tensor]] = list()

        # Start by taking the first partial’s order:
        order: int = partials[0].order
        pointer: int = order * (1 + int(batch)) + 1
        indices: list[int] = list(range(pointer))
        ct_indices: list[int] = indices[len(indices) - order * (1 + int(batch)) :]
        nc_indices: list[int] = [idx for idx in indices if idx not in ct_indices]

        # Add the first pretensor + index mapping
        pretensor: Tensor = pretensors[order - 1]
        einsum_args.append(pretensor)
        einsum_args.append(indices)

        if batch:
            assert pretensor.ndim == 1 + 2 * order
        else:
            assert pretensor.ndim == 1 + order

        # Now handle each subsequent partial
        for p_idx, p in enumerate(partials[1:]):

            if batch:

                # construct identity of shape
                n: int = pretensor.shape[2 * p_idx + 1]
                dims: int = p.order + 1
                identity: Tensor = construct_nd_identity(n=n, dims=dims)

                # calculate indices & update pointer
                new_pointer: int = pointer + p.order
                batch_new_nc_indices: list[int] = list(range(pointer, new_pointer))
                pointer = new_pointer

                # define the identity indices
                bridge: int = ct_indices.pop(0)
                indices = [bridge, *batch_new_nc_indices]

                # # aggregate the identity tensor to the einsum arguments
                einsum_args.append(identity)
                einsum_args.append(indices)

            # calculate indices & update pointer
            new_pointer: int = pointer + p.order
            new_nc_indices: list[int] = list(range(pointer, new_pointer))
            pointer = new_pointer

            # Take one index from ct_indices to form the new "bridge"
            bridge: int = ct_indices.pop(0)
            indices = [bridge, *new_nc_indices]

            # aggregate the identity tensor to the einsum arguments
            subtensor: Tensor = subtensors[p.order - 1]
            einsum_args.append(subtensor)
            einsum_args.append(indices)

            if batch:
                assert len(new_nc_indices) == len(batch_new_nc_indices)
                for i, idx in enumerate(new_nc_indices):
                    nc_indices.append(batch_new_nc_indices[i])
                    nc_indices.append(idx)
            else:
                nc_indices.extend(new_nc_indices)

        # By now, we should have consumed all ct_indices
        assert len(ct_indices) == 0, "Some contracting indices were not used."

        # Perform the actual contraction
        einsum_args.append(nc_indices)
        result_tensor: Tensor = einsum(*einsum_args)

        # Scale by product coefficient
        if product.coefficient != 1:
            result_tensor = product.coefficient * result_tensor

        # Accumulate
        if accum is None:
            accum = result_tensor
        else:
            if accum.shape != result_tensor.shape:
                raise ValueError(
                    "All partial contractions must have the same shape. "
                    f"Current accum shape={accum.shape}, "
                    f"new tensor shape={result_tensor.shape}"
                )
            accum += result_tensor

    if accum is None:
        raise RuntimeError("No products found in the expression, nothing to contract.")

    return accum
