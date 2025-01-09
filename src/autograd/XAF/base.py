# python 3.12

# Standard Library dependencies
import gc
from abc import ABC, abstractmethod
from typing import Any, Tuple, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from automatic_derivation import calculate_n_order_partial, SumGroup
from contractor import contractor
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

    def _update_multipartials(self, multipartials: list[list[Tensor]], shapes: list[Tuple[int, ...]]) -> None:

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

    @abstractmethod
    def _get_context(self) -> Any:
        # extract the ctx variables from the grad_fn attributes
        # must be called once the class is integral
        pass

    @abstractmethod
    def _differentiation(self, shaped_output_partials: ShapedPartials, idx: int) -> None:
        # return boolean indecates whether:
        #   the partials are finished (True)
        #   the partials require more calls to be finished (False)
        pass