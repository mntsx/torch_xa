# python 3.12

# Standard Library dependencies
from abc import ABC, abstractmethod
from typing import Tuple, Type, Union

# Internal dependencies
from src.autograd.configurations.selectors import Selector, XAFselector
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction


class XAFexchanger(ABC):
    _selector: Union[None, Selector] = None

    @classmethod
    def _select(
        cls, grad_fn: AutogradFunction, order: int
    ) -> Type[ExtendedAutogradFunction]:
        assert cls._selector is not None, "Selector is not set."
        XAF: Type[ExtendedAutogradFunction] = cls._selector(
            grad_fn=grad_fn, order=order
        )
        if isinstance(XAF, cls.targets()):
            XAF = cls.transform(XAF)
        assert issubclass(type(XAF), ExtendedAutogradFunction)
        return XAF

    @classmethod
    def get_selector(cls) -> Union[None, Selector]:
        return cls._select

    @classmethod
    def set_selector(cls, value: Selector) -> None:
        assert issubclass(type(value), (Selector, XAFselector))
        cls._selector = value

    @classmethod
    @abstractmethod
    def targets(cls) -> Tuple[Type[ExtendedAutogradFunction], ...]:
        # Returns the ExtendedAutogradFunction classes that will get transformed
        pass

    @classmethod
    @abstractmethod
    def transform(
        cls, base_XAF: Type[ExtendedAutogradFunction]
    ) -> Type[ExtendedAutogradFunction]:
        # Transforms a subset of ExtendedAutogradFunction classes into new ones
        pass
