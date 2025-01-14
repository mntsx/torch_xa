import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.interfaces import backward, Superset

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_1() -> None:
    T0: Tensor = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O: Tensor = torch.mm(T0, T1)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())


def test_2() -> None:

    T0: Tensor = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T1: Tensor = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O: Tensor = torch.mm(T0, T1)
    superset: Superset = Superset.construct(source=O)
    superset.backward(order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())


def test_3() -> None:

    T0: Tensor
    T1: Tensor
    O: Tensor

    T0 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.add(T0, T1)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())

    T0 = torch.rand(size=(1, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.add(T0, T1)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())

    T0 = torch.rand(size=(4, 1), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.add(T0, T1)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())


def test_4() -> None:

    T0: Tensor
    T1: Tensor
    O: Tensor

    T0 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.mul(T0, T1)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())

    T0 = torch.rand(size=(1, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.mul(T0, T1)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())

    T0 = torch.rand(size=(4, 1), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O = torch.mul(T0, T1)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())


def test_5() -> None:

    T0: Tensor
    T1: Tensor
    T2: Tensor
    O: Tensor

    T0 = torch.rand(size=(4, 8), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T2 = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O = torch.addmm(input=T0, mat1=T1, mat2=T2)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())
    assert T2.xgrad[0].shape == (O.numel(), T2.numel())

    T0 = torch.rand(size=(8,), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T2 = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O = torch.addmm(input=T0, mat1=T1, mat2=T2)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())
    assert T2.xgrad[0].shape == (O.numel(), T2.numel())

    T0 = torch.rand(size=(1,), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T2 = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O = torch.addmm(input=T0, mat1=T1, mat2=T2)
    backward(source=O, order=2)
    assert T0.xgrad[0].shape == (O.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O.numel(), T1.numel())
    assert T2.xgrad[0].shape == (O.numel(), T2.numel())


def test_6() -> None:

    T: Tensor = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O: Tensor = torch.relu(T)
    backward(source=O, order=2)
    assert T.xgrad[0].shape == (O.numel(), T.numel())


def test_7() -> None:

    T: Tensor = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O: Tensor = torch.sigmoid(T)
    backward(source=O, order=2)
    assert T.xgrad[0].shape == (O.numel(), T.numel())


def test_8() -> None:

    T: Tensor = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O: Tensor = torch.tanh(T)
    backward(source=O, order=2)
    assert T.xgrad[0].shape == (O.numel(), T.numel())


def test_9() -> None:

    T: Tensor = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O: Tensor = torch.softmax(T, dim=1)
    backward(source=O, order=2)
    assert T.xgrad[0].shape == (O.numel(), T.numel())


def test_10() -> None:

    T: Tensor = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O: Tensor = T.t()
    backward(source=O, order=2)
    assert T.xgrad[0].shape == (O.numel(), T.numel())


def test_11() -> None:

    T: Tensor = torch.rand(size=(2, 3, 4, 5), requires_grad=True, device=device)
    O: Tensor = T.transpose(dim0=0, dim1=2)
    backward(source=O, order=2)
    assert T.xgrad[0].shape == (O.numel(), T.numel())
