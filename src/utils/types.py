# python 3.12

# Standard Library dependencies
from typing import Tuple, Type

# PyTorch dependencies
import torch
from torch import Tensor


type AutogradFunction = Type[torch.autograd.Function]
type Partials = Tuple[Tensor, ...]
type ShapedPartials = Tuple[Partials, Tuple[int, ...]]
type MultiPartials = Tuple[ShapedPartials, ...]
# type Selector = callable[[AutogradFunction, int], Type[ExtendedAutogradFunction]]
