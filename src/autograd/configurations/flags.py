# python 3.12

# Standard Library dependencies
from abc import ABC
from typing import Any


class Flag(ABC):

    @staticmethod
    def process_values(*args: Any) -> Any:
        pass
