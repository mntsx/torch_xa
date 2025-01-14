# Standard Library dependencies
from typing import Tuple

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.backprop import contractor, hadamard
from src.autograd.engine.symbolic.derivation import calculate_n_order_partial, SumGroup

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

A: int = 2
B: int = 4
X: int = 6
Bch: int = 8


def test_no_batch_order_2_contraction() -> None:
    partialA1: Tensor = torch.ones(size=(A, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, X, X), device=device)
    partialB1: Tensor = torch.ones(size=(X, B), device=device)
    partialB2: Tensor = torch.ones(size=(X, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=2)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(False, False),
    )
    assert tuple(contracted_tensor.shape) == (A, B, B)


def test_no_batch_order_3_contraction() -> None:
    partialA1: Tensor = torch.ones(size=(A, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, X, X), device=device)
    partialA3: Tensor = torch.ones(size=(A, X, X, X), device=device)
    partialB1: Tensor = torch.ones(size=(X, B), device=device)
    partialB2: Tensor = torch.ones(size=(X, B, B), device=device)
    partialB3: Tensor = torch.ones(size=(X, B, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=3)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2, partialA3)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2, partialB3)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(False, False),
    )
    assert tuple(contracted_tensor.shape) == (A, B, B, B)


def test_pre_batched_order_2_contraction() -> None:
    partialA1: Tensor = torch.ones(size=(A, Bch, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, Bch, X, Bch, X), device=device)
    partialB1: Tensor = torch.ones(size=(X, B), device=device)
    partialB2: Tensor = torch.ones(size=(X, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=2)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(True, False),
    )
    assert tuple(contracted_tensor.shape) == (A, Bch, B, Bch, B)


def test_pre_batched_order_3_contraction() -> None:
    partialA1: Tensor = torch.ones(size=(A, Bch, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, Bch, X, Bch, X), device=device)
    partialA3: Tensor = torch.ones(size=(A, Bch, X, Bch, X, Bch, X), device=device)
    partialB1: Tensor = torch.ones(size=(X, B), device=device)
    partialB2: Tensor = torch.ones(size=(X, B, B), device=device)
    partialB3: Tensor = torch.ones(size=(X, B, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=3)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2, partialA3)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2, partialB3)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(True, False),
    )
    assert tuple(contracted_tensor.shape) == (A, Bch, B, Bch, B, Bch, B)


def test_post_batched_order_2_contraction() -> None:

    partialA1: Tensor = torch.ones(size=(A, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, X, X), device=device)
    partialB1: Tensor = torch.ones(size=(Bch, X, B), device=device)
    partialB2: Tensor = torch.ones(size=(Bch, X, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=2)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(False, True),
    )
    assert tuple(contracted_tensor.shape) == (A, Bch, B, Bch, B)


def test_post_batched_order_3_contraction() -> None:

    partialA1: Tensor = torch.ones(size=(A, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, X, X), device=device)
    partialA3: Tensor = torch.ones(size=(A, X, X, X), device=device)
    partialB1: Tensor = torch.ones(size=(Bch, X, B), device=device)
    partialB2: Tensor = torch.ones(size=(Bch, X, B, B), device=device)
    partialB3: Tensor = torch.ones(size=(Bch, X, B, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=3)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2, partialA3)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2, partialB3)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(False, True),
    )
    assert tuple(contracted_tensor.shape) == (A, Bch, B, Bch, B, Bch, B)


def test_full_batched_order_2_contraction() -> None:

    partialA1: Tensor = torch.ones(size=(A, Bch, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, Bch, X, Bch, X), device=device)
    partialB1: Tensor = torch.ones(size=(Bch, X, B), device=device)
    partialB2: Tensor = torch.ones(size=(Bch, X, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=2)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(True, True),
    )
    assert tuple(contracted_tensor.shape) == (A, Bch, B, Bch, B)


def test_full_batched_order_3_contraction() -> None:

    partialA1: Tensor = torch.ones(size=(A, Bch, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, Bch, X, Bch, X), device=device)
    partialA3: Tensor = torch.ones(size=(A, Bch, X, Bch, X, Bch, X), device=device)
    partialB1: Tensor = torch.ones(size=(Bch, X, B), device=device)
    partialB2: Tensor = torch.ones(size=(Bch, X, B, B), device=device)
    partialB3: Tensor = torch.ones(size=(Bch, X, B, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=3)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2, partialA3)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2, partialB3)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(True, True),
    )
    assert tuple(contracted_tensor.shape) == (A, Bch, B, Bch, B, Bch, B)


def test_redundant_tensors_contraction() -> None:

    partialA1: Tensor = torch.ones(size=(A, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, X, X), device=device)
    partialA3: Tensor = torch.ones(size=(A, X, X, X), device=device)
    partialB1: Tensor = torch.ones(size=(X, B), device=device)
    partialB2: Tensor = torch.ones(size=(X, B, B), device=device)
    partialB3: Tensor = torch.ones(size=(X, B, B, B), device=device)

    expression: SumGroup = calculate_n_order_partial(n=2)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2, partialA3)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2, partialB3)
    contracted_tensor: Tensor = contractor(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
        batch=(False, False),
    )
    assert tuple(contracted_tensor.shape) == (A, B, B)


def test_order_2_hadamard() -> None:
    partialA1: Tensor = torch.ones(size=(A, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, X, X), device=device)
    partialB1: Tensor = torch.ones(size=(X,), device=device)
    partialB2: Tensor = torch.ones(size=(X,), device=device)

    expression: SumGroup = calculate_n_order_partial(n=2)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2)
    contracted_tensor: Tensor = hadamard(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
    )
    assert tuple(contracted_tensor.shape) == (A, X, X)


def test_order_3_hadamard() -> None:
    partialA1: Tensor = torch.ones(size=(A, X), device=device)
    partialA2: Tensor = torch.ones(size=(A, X, X), device=device)
    partialA3: Tensor = torch.ones(size=(A, X, X, X), device=device)
    partialB1: Tensor = torch.ones(size=(X,), device=device)
    partialB2: Tensor = torch.ones(size=(X,), device=device)
    partialB3: Tensor = torch.ones(size=(X,), device=device)

    expression: SumGroup = calculate_n_order_partial(n=3)

    pretensors: Tuple[Tensor, ...] = (partialA1, partialA2, partialA3)
    subtensors: Tuple[Tensor, ...] = (partialB1, partialB2, partialB3)
    contracted_tensor: Tensor = hadamard(
        pretensors=pretensors,
        subtensors=subtensors,
        expression=expression,
    )
    assert tuple(contracted_tensor.shape) == (A, X, X, X)
