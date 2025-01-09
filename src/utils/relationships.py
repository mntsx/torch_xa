# python 3.12

# Standard Library dependencies
import gc
from typing import Type, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
import src.autograd.XAF as XAF
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.types import AutogradFunction


def grad_fn_map(grad_fn: AutogradFunction) -> Type:

    TA: Tensor = torch.zeros(size=(1,), requires_grad=True)
    TB: Tensor = torch.zeros(size=(1, 1), requires_grad=True)
    XAF_class: Union[None, Type[ExtendedAutogradFunction]] = None

    aux: Tensor

    # ACCUMULATION

    if type(grad_fn) is type(torch.sum(TA).grad_fn.next_functions[0][0]):
        XAF_class = XAF.AccumulateGradX

    # RESHAPE

    # torch.view, torch.Tensor.view
    aux = TA.view(size=(1, 1))
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.ViewXBackward0

    # torch.t, torch.Tensor.t
    aux = torch.t(input=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.TXBackward0

    # torch.transpose, torch.Tensor.transpose
    aux = torch.transpose(input=TB, dim0=0, dim1=1)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.TransposeXBackward0

    # torch.permute, torch.Tensor.permute, torch.Tensor.T
    aux = torch.permute(input=TB, dims=(0, 1))
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.PermuteXBackward0

    # TENSOR OPERATIONS

    # tensor hadamard product, tensor scalar product, tensor sum

    # MATRIX MUTIPLICATION

    # torch.addmm, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.addmm(input=TB, mat1=TB, mat2=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.AddmmXBackward0

    # @, torch.mm, torch.matmul, torch.nn.Linear, torch.nn.functional.linear
    aux = torch.mm(input=TB, mat2=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.MmXBackward0  # XAF.linalg.mm.MmXBackward0

    # ACTIVATIONS

    # torch.nn.ReLU, torch.nn.functional.relu
    aux = torch.nn.functional.relu(TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.ReluXBackward0

    # torch.nn.Sigmoid, torch.nn.functional.sigmoid
    aux = torch.nn.functional.sigmoid(TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SigmoidXBackward0

    # torch.nn.Tanh, torch.nn.functional.tanh
    aux = torch.nn.functional.tanh(TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.TanhXBackward0

    # torch.nn.Softmax, torch.nn.functional.softmax
    aux = torch.nn.functional.softmax(TA, dim=0)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SoftmaxBackward0

    del aux
    gc.collect()

    if grad_fn is not None and XAF_class is None:
        raise NotImplementedError(f"{grad_fn.name()} is not supported.")

    return XAF_class
