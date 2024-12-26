# Standard Library dependencies
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple, Type, Union

# PyTorch dependencies
import torch
from torch import Tensor


class GeneralizedBackward:

	def __init__(self, grad_fn: Type[torch.autograd.Function]) -> None:
		self._grad_fn = grad_fn

	def __call__(self, input: Union[Tensor, Tuple[Tensor]], n: Optional[int]) -> Union[Tensor, Tuple[Tensor]]:
		
		if n is None:
			output = self._grad_fn(input)
		else:
			assert isinstance(n, int)
			output = _high_order_backward(input)


	def _high_order_backward() -> Union[Tensor, Tuple[Tensor]]:
		pass