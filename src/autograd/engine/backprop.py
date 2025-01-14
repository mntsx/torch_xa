# python 3.12

# Standard Library dependencies
from typing import Optional, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.symbolic.derivation import SumGroup, ProductGroup, Partial
from src.tensors.functional import construct_nd_identity, einsum


def contractor(
    pretensors: Tuple[Tensor, ...],
    subtensors: Tuple[Tensor, ...],
    expression: SumGroup,
    device: torch.device,
    batch: Optional[Tuple[bool, bool]] = (False, False),
) -> Tensor:

    bb: bool = any(batch)
    expression.ensure_sorted()
    accum: Optional[Tensor] = None
    identity: Tensor

    pre_non_zero: list[bool] = [not (T.max() == 0 and T.min() == 0) for T in pretensors]
    sub_non_zero: list[bool] = [not (T.max() == 0 and T.min() == 0) for T in subtensors]
    non_zero_products: list[ProductGroup] = list()

    for product in expression.products:
        partials: list[Partial] = product.partials
        non_zero: bool = pre_non_zero[(partials[0].order - 1)]
        for p in partials[1:]:
            non_zero = non_zero and sub_non_zero[(p.order - 1)]
        if non_zero:
            non_zero_products.append(product)

    for product in non_zero_products:
        product.ensure_sorted()
        partials: list[Partial] = product.partials

        # Collect arguments for the einsum call
        einsum_args: list[Union[list[int], Tensor]] = list()

        # Start by taking the first partial’s order:
        order: int = partials[0].order
        pointer: int = order * (1 + int(bb)) + 1
        indices: list[int] = list(range(pointer))
        ct_indices: list[int] = indices[len(indices) - order * (1 + int(bb)) :]
        nc_indices: list[int] = [idx for idx in indices if idx not in ct_indices]

        # Add the first pretensor + index mapping
        pretensor: Tensor = pretensors[order - 1]
        if bb and not batch[0]:
            assert pretensors[order - 1].ndim == 1 + order
            T: Tuple[int, ...] = tuple(range(order + 1))
            I: Tuple[int, ...] = tuple(range(order + 1, 2 * order + 1))
            TI: Tuple[int, ...] = (*[i for ii in zip(T, I) for i in ii], T[-1])
            batch_size: int = subtensors[0].shape[0]
            identity = construct_nd_identity(n=batch_size, dims=order, device=device)
            pretensor.ndim == 1 + order
            pretensor = torch.einsum(pretensor, T, identity, I, TI)
        assert pretensor.ndim == 1 + order * (1 + int(bb))

        einsum_args.append(pretensor)
        einsum_args.append(indices)

        # Now handle each subsequent partial
        for p_idx, p in enumerate(partials[1:]):

            if bb:

                # construct identity of shape
                n: int = pretensor.shape[2 * p_idx + 1]
                dims: int = p.order + 1
                identity = construct_nd_identity(n=n, dims=dims, device=device)

                # calculate new indices & update pointer
                new_pointer: int = pointer + p.order
                batch_new_nc_indices: list[int] = list(range(pointer, new_pointer))
                pointer = new_pointer

                # define the identity indices
                ibridge: int = ct_indices.pop(0)
                indices = [ibridge, *batch_new_nc_indices]

                # aggregate the identity tensor to the einsum arguments
                einsum_args.append(identity)
                einsum_args.append(indices)

            # calculate indices & update pointer
            new_pointer: int = pointer + p.order
            new_nc_indices: list[int] = list(range(pointer, new_pointer))
            pointer = new_pointer

            # Take one index from ct_indices to form the new "bridge"
            bridge: int = ct_indices.pop(0)
            indices = [ibridge] if batch[1] else []
            indices.extend([bridge, *new_nc_indices])

            # aggregate the tensor to the einsum arguments
            subtensor: Tensor = subtensors[p.order - 1]
            einsum_args.append(subtensor)
            einsum_args.append(indices)

            if bb:
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
        if len(non_zero_products) == 0:
            mx: int = max([p.order for P in expression.products for p in P.partials])
            dim0_size: int = pretensors[0].shape[0]
            batch_size: Tuple[int] = tuple()
            if batch[0]:
                batch_size = (pretensors[0].shape[1],)
            elif batch[1]:
                batch_size = (subtensors[0].shape[0],)
            output_size: int = subtensors[0].shape[-1]
            shape: Tuple[int, ...] = (dim0_size, *(mx * (*batch_size, output_size)))
            accum = torch.zeros(size=shape, device=device)
        else:
            raise RuntimeError(
                "No products found in the expression, nothing to contract."
            )

    return accum


def hadamard(
    pretensors: Tuple[Tensor, ...],
    subtensors: Tuple[Tensor, ...],
    expression: SumGroup,
    device: torch.device,
) -> Tensor:

    assert all(T.ndim == 1 for T in subtensors)

    expression.ensure_sorted()
    accum: Optional[Tensor] = None

    pre_non_zero: list[bool] = [not (T.max() == 0 and T.min() == 0) for T in pretensors]
    sub_non_zero: list[bool] = [not (T.max() == 0 and T.min() == 0) for T in subtensors]
    non_zero_products: list[ProductGroup] = list()

    for product in expression.products:
        partials: list[Partial] = product.partials
        non_zero: bool = pre_non_zero[(partials[0].order - 1)]
        for p in partials[1:]:
            non_zero = non_zero and sub_non_zero[(p.order - 1)]
        if non_zero:
            non_zero_products.append(product)

    for product in non_zero_products:

        product.ensure_sorted()
        partials: list[Partial] = product.partials

        # Collect arguments for the einsum call
        einsum_args: list[Union[list[int], Tensor]] = list()

        # Start by taking the first partial’s order:
        order: int = partials[0].order
        pointer: int = order + 1
        indices: list[int] = list(range(pointer))
        hadamard_indices: list[int] = indices[1:]  # hadamard indides
        final_indices: list[int] = [indices[0]]

        # Add the first pretensor + index mapping
        pretensor: Tensor = pretensors[order - 1]
        einsum_args.append(pretensor)
        einsum_args.append(indices)

        # Now handle each subsequent partial
        for p in partials[1:]:

            # aggregate subtensor to the einsum arguments
            bridge: int = hadamard_indices.pop(0)
            indices = [bridge]
            subtensor: Tensor = subtensors[p.order - 1]
            final_indices.append(bridge)

            einsum_args.append(subtensor)
            einsum_args.append(indices)

            if p.order > 1:

                # Create identity tensor
                n: int = subtensor.numel()
                identity: Tensor = construct_nd_identity(
                    n=n, dims=p.order, device=device
                )

                # calculate new indices & update pointer
                new_pointer: int = pointer + (p.order - 1)
                new_indices: list[int] = list(range(pointer, new_pointer))
                pointer = new_pointer

                # Define identity indices
                indices = [bridge, *new_indices]
                final_indices.extend(new_indices)

                # aggregate the identity tensor to the einsum arguments
                einsum_args.append(identity)
                einsum_args.append(indices)

        # By now, we should have consumed all hadamard indices
        assert len(hadamard_indices) == 0, "Some contracting indices were not used."

        # Perform the actual contraction
        einsum_args.append(final_indices)
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
                    "All partial hadamard multiplications must have the same shape. "
                    f"Current accum shape={accum.shape}, "
                    f"new tensor shape={result_tensor.shape}"
                )
            accum += result_tensor

    if accum is None:
        if len(non_zero_products) == 0:
            mx: int = max([p.order for P in expression.products for p in P.partials])
            dim0_size: int = pretensors[0].shape[0]
            output_size: int = subtensors[0].shape[0]
            shape: Tuple[int, ...] = (
                dim0_size,
                *(mx * (output_size,)),
            )
            accum = torch.zeros(size=shape, device=device)
        else:
            raise RuntimeError(
                "No products found in the expression, nothing to contract."
            )

    return accum
