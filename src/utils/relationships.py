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

    # CONDITIONALS

    # *, torch.mul, torch.multiply, torch.Tensor.mul
    aux = torch.where(condition=TB > 0, input=TB, other=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.WhereXBackward0

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

    # POWERS

    # *, torch.mul, torch.multiply, torch.Tensor.mul
    aux = torch.pow(input=TB, exponent=2)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.PowXBackward0

    # *, torch.mul, torch.multiply, torch.Tensor.mul
    aux = torch.pow(input=TB, exponent=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.PowXBackward1

    # torch.prod, torch.Tensor.prod
    aux = torch.sqrt(input=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SqrtXBackward0

    # torch.prod, torch.Tensor.prod
    aux = torch.exp(input=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.ExpXBackward0

    # PRODUCTS

    # *, torch.mul, torch.multiply, torch.Tensor.mul
    aux = torch.div(input=TB, other=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.DivXBackward0

    # *, torch.mul, torch.multiply, torch.Tensor.mul
    aux = torch.mul(input=TB, other=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.MulXBackward0

    # torch.prod, torch.Tensor.prod
    aux = torch.prod(input=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.ProdXBackward0

    # torch.prod, torch.Tensor.prod
    aux = torch.prod(input=TB, dim=1)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.ProdXBackward1

    # SUMMATIONS

    # +, torch.add, torch.Tensor.add
    aux = torch.sub(input=TB, other=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SubXBackward0

    # -, torch.sub, torch.Tensor.sub
    aux = torch.add(input=TB, other=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.AddXBackward0

    # torch.sum, torch.Tensor.sum
    aux = torch.sum(input=TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SumXBackward0

    # torch.sum, torch.Tensor.sum
    aux = torch.sum(input=TB, dim=(1,))
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SumXBackward1

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

    # torch.nn.ELU, torch.nn.functional.elu
    aux = torch.nn.functional.elu(input=TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.EluXBackward0

    # torch.nn.LeakyReLU, torch.nn.functional.leaky_relu
    aux = torch.nn.functional.leaky_relu(input=TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.LeakyReluXBackward0

    # torch.nn.PreLU, torch.nn.functional.prelu
    aux = torch.nn.functional.prelu(input=TA, weight=TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.PreluKernelXBackward0

    # torch.nn.ReLU, torch.nn.functional.relu
    aux = torch.nn.functional.relu(input=TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.ReluXBackward0

    # torch.nn.SeLU, torch.nn.functional.selu
    aux = torch.nn.functional.selu(input=TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SeluXBackward0

    # torch.nn.Sigmoid, torch.nn.functional.sigmoid
    aux = torch.nn.functional.sigmoid(input=TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SigmoidXBackward0

    # torch.nn.Tanh, torch.nn.functional.tanh
    aux = torch.nn.functional.tanh(input=TA)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.TanhXBackward0

    # torch.nn.Softmax, torch.nn.functional.softmax
    aux = torch.nn.functional.softmax(input=TA, dim=0)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SoftmaxBackward0

    # OTHER MATHEMATICAL OPERATORS

    aux = torch.cos(TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.CosXBackward0

    aux = torch.log(TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.LogXBackward0

    aux = torch.sin(TB)
    if type(grad_fn) is type(aux.grad_fn):
        XAF_class = XAF.SinXBackward0

    del aux
    gc.collect()

    if grad_fn is not None and XAF_class is None:
        raise NotImplementedError(f"{grad_fn.name()} is not supported.")

    return XAF_class
