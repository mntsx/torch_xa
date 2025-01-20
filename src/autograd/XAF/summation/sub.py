# python 3.12

# Standard Library dependencies
import torch
from typing import Tuple

# Internal dependencies
from src.autograd.XAF.summation.add import AddXBackward0
from src.utils.types import AutogradFunction


class SubXBackward0(AddXBackward0):

    def __init__(
        self, grad_fn: AutogradFunction, order: int, device: torch.device
    ) -> None:
        super().__init__(grad_fn=grad_fn, order=order, device=device)
        return None

    def _get_context(self) -> Tuple[float]:
        saved_alpha: float = -self.grad_fn._saved_alpha
        return (saved_alpha,)
