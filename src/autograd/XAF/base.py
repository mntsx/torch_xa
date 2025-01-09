# python 3.12

# Standard Library dependencies
import gc
import math
import warnings
from abc import ABC, abstractmethod
from itertools import permutations
from typing import Any, Tuple, Union

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from src.utils.partials import unbroadcast_partials
from src.utils.types import AutogradFunction, MultiPartials, Partials, ShapedPartials


class ExtendedAutogradFunction(ABC):

    # equivalent to grad_fn for arbitrary dimension
    # for each expected input (each input = up-to-n order partial
    # derivates of computational graph output tensor with respect to
    # one of this function's associated layer outputs) computes and
    # accumulates (through sumation) the up-to-n order partial
    # derivatives of computational graph output tensor with respect to
    # this function's associated layer input.

    # Optionally, it can also compute the partials with respect to the
    # layer parameters

    def __init__(self, grad_fn: AutogradFunction, order: int) -> None:
        self._grad_fn: AutogradFunction = grad_fn
        self._order: int = order

        self._retain_partials: bool = False
        self._multipartials: Union[None, MultiPartials] = None

        self._output_registry: list[int] = []
        self._received_calls: list[int] = []

        return None

    def __call__(self, shaped_output_partials: ShapedPartials, idx: int) -> bool:
        assert idx in self._output_registry
        assert idx not in self._received_calls
        assert len(shaped_output_partials[0]) == self._order
        self._received_calls.append(idx)
        self._received_calls.sort()
        self._differentiation(shaped_output_partials=shaped_output_partials, idx=idx)
        return self.available

    def register_idx(self, idx: int) -> None:
        # registers the index of expected new group of output partials
        assert idx not in self._output_registry
        self._output_registry.append(idx)
        self._output_registry.sort()
        return None

    @property
    def grad_fn(self) -> AutogradFunction:
        return self._grad_fn

    @property
    def partials(self) -> MultiPartials:
        return self._multipartials

    @property
    def available(self) -> bool:
        # indicates whether all specified inputs have ben received
        aux: list[bool] = [r in self._received_calls for r in self._output_registry]
        all_received: bool = all(aux)
        return all_received

    @property
    def retain_partials(self) -> bool:
        return self._retain_partials

    @retain_partials.setter
    def retain_partials(self, value: bool) -> None:
        self._retain_partials = value
        return None

    @property
    @abstractmethod
    def integral(self) -> bool:
        # determine if all the required inputs have been registered
        pass

    def clear_partials(self) -> None:
        if not self._retain_partials:
            self._multipartials = None
        gc.collect()
        return None

    def _update_multipartials(
        self, multipartials: list[list[Tensor]], shapes: list[Tuple[int, ...]]
    ) -> None:

        assert len(multipartials) == len(shapes)

        if self._multipartials is not None:
            for i, _ in enumerate(multipartials):
                for j, _ in multipartials[i]:
                    assert multipartials[i][j].shape == self._multipartials[i].shape
                    multipartials[i][j] += self._multipartials[i]
                assert shapes[i] == self._multipartials[i][1]

        aux: list[ShapedPartials] = list()
        for i, _ in enumerate(multipartials):
            partials: Partials = tuple(multipartials[i])
            shape: Tuple[int, ...] = shapes[i]
            aux.append((partials, shape))
        self._multipartials = tuple(aux)

        return None

    def _unbroadcast_partials(
        self, shaped_partials: ShapedPartials, output_shape: Tuple[int, ...]
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
            new_shaped_partials = unbroadcast_partials(
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
                    "XAF found an intractable combination of permutation "
                    "and broadcasting. Consider being more explicit in "
                    "the arrangement of dimensions."
                )

            if len(broadcastable_perms) > 1:
                warnings.warn(
                    "XAF found an ambiguous combination of permutation "
                    "and broadcasting. This can lead to errors in partials "
                    "computations. Consider being more explicit in the "
                    "arrangement of dimensions.",
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
            new_shaped_partials = unbroadcast_partials(
                shaped_partials=shaped_partials, output_shape=best_permutation
            )

        else:

            raise ValueError(
                "XAF found an intractable combination of permutation "
                "and broadcasting. Consider being more explicit in "
                "the arrangement of dimensions."
            )

        return new_shaped_partials

    @abstractmethod
    def _get_context(self) -> Any:
        # extract the ctx variables from the grad_fn attributes
        # must be called once the class is integral
        pass

    @abstractmethod
    def _differentiation(
        self, shaped_output_partials: ShapedPartials, idx: int
    ) -> None:
        # return boolean indecates whether:
        #   the partials are finished (True)
        #   the partials require more calls to be finished (False)
        pass
