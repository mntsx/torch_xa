# python 3.12

# Standard Library dependencies
from abc import ABC, abstractmethod
from typing import Type

# Internal dependencies
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.autograd.XAF.testing import TestXBackward0
from src.utils.relationships import grad_fn_map
from src.utils.types import AutogradFunction


type Selector = callable[[AutogradFunction, int], Type[ExtendedAutogradFunction]]


class XAFselector(ABC):

    @classmethod
    def _select(
        cls, grad_fn: AutogradFunction, order: int
    ) -> Type[ExtendedAutogradFunction]:
        XAF: Type[ExtendedAutogradFunction] = cls.select(grad_fn=grad_fn, order=order)
        assert issubclass(type(XAF), ExtendedAutogradFunction)
        return XAF

    @classmethod
    @abstractmethod
    def select(
        cls, grad_fn: AutogradFunction, order: int
    ) -> Type[ExtendedAutogradFunction]:
        pass


class DefaultSelector(XAFselector):

    @classmethod
    def select(
        cls, grad_fn: AutogradFunction, order: int
    ) -> Type[ExtendedAutogradFunction]:
        XAF_class: Type = grad_fn_map(grad_fn=grad_fn)
        XAF: Type[ExtendedAutogradFunction] = XAF_class(grad_fn=grad_fn, order=order)
        return XAF


class TestSelector(XAFselector):

    @classmethod
    def select(
        cls, grad_fn: AutogradFunction, order: int
    ) -> Type[ExtendedAutogradFunction]:
        extended_backward: Type[ExtendedAutogradFunction]
        extended_backward = TestXBackward0(grad_fn=grad_fn, order=order)
        return extended_backward
