import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.interfaces import backward

device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_1() -> None:
    T0: Tensor = torch.rand(size=(8,), requires_grad=True)
    T1: Tensor = torch.rand(size=(4, 6), requires_grad=True)
    T2: Tensor = torch.rand(size=(6, 8), requires_grad=True)
    O0: Tensor = torch.addmm(input=T0, mat1=T1, mat2=T2)
    O0 = O0.sum()

    backward(source=O0, order=2)
    O0.backward()

    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())
    assert torch.allclose(T0.xgrad[0].flatten(), T0.grad.flatten())
    assert torch.allclose(T1.xgrad[0].flatten(), T1.grad.flatten())


def test_2() -> None:

    T0: Tensor = torch.rand(size=(4, 6), requires_grad=True)
    T1: Tensor = torch.rand(size=(6, 8), requires_grad=True)
    O0: Tensor = torch.mm(T0, T1)
    O0 = O0.sum()

    backward(source=O0, order=2)
    O0.backward()

    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())
    assert torch.allclose(T0.xgrad[0].flatten(), T0.grad.flatten())
    assert torch.allclose(T1.xgrad[0].flatten(), T1.grad.flatten())


def test_3() -> None:

    T0: Tensor
    T1: Tensor

    T0 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O0: Tensor = torch.add(T0, T1)
    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())

    T0 = torch.rand(size=(1, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O1: Tensor = torch.add(T0, T1)
    backward(source=O1, order=2)
    assert T0.xgrad[0].shape == (O1.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O1.numel(), T1.numel())

    T0 = torch.rand(size=(4, 1), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O2: Tensor = torch.add(T0, T1)
    backward(source=O2, order=2)
    assert T0.xgrad[0].shape == (O2.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O2.numel(), T1.numel())


def test_3b() -> None:

    T0: Tensor
    T1: Tensor

    T0 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O0: Tensor = torch.sub(T0, T1)
    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())

    T0 = torch.rand(size=(1, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O1: Tensor = torch.sub(T0, T1)
    backward(source=O1, order=2)
    assert T0.xgrad[0].shape == (O1.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O1.numel(), T1.numel())

    T0 = torch.rand(size=(4, 1), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O2: Tensor = torch.sub(T0, T1)
    backward(source=O2, order=2)
    assert T0.xgrad[0].shape == (O2.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O2.numel(), T1.numel())


def test_4() -> None:

    T0: Tensor
    T1: Tensor

    T0 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O0: Tensor = torch.mul(T0, T1)
    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())

    T0 = torch.rand(size=(1, 6), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O1: Tensor = torch.mul(T0, T1)
    backward(source=O1, order=2)
    assert T0.xgrad[0].shape == (O1.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O1.numel(), T1.numel())

    T0 = torch.rand(size=(4, 1), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    O2 = torch.mul(T0, T1)
    backward(source=O2, order=2)
    assert T0.xgrad[0].shape == (O2.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O2.numel(), T1.numel())


def test_5() -> None:

    T0: Tensor
    T1: Tensor
    T2: Tensor

    T0 = torch.rand(size=(4, 8), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T2 = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O0: Tensor = torch.addmm(input=T0, mat1=T1, mat2=T2)
    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())
    assert T2.xgrad[0].shape == (O0.numel(), T2.numel())

    T0 = torch.rand(size=(8,), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T2 = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O1: Tensor = torch.addmm(input=T0, mat1=T1, mat2=T2)
    backward(source=O1, order=2)
    assert T0.xgrad[0].shape == (O1.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O1.numel(), T1.numel())
    assert T2.xgrad[0].shape == (O1.numel(), T2.numel())

    T0 = torch.rand(size=(1,), requires_grad=True, device=device)
    T1 = torch.rand(size=(4, 6), requires_grad=True, device=device)
    T2 = torch.rand(size=(6, 8), requires_grad=True, device=device)
    O2: Tensor = torch.addmm(input=T0, mat1=T1, mat2=T2)
    backward(source=O2, order=2)
    assert T0.xgrad[0].shape == (O2.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O2.numel(), T1.numel())
    assert T2.xgrad[0].shape == (O2.numel(), T2.numel())


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


def test_12() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = T0.sum()
    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    O0.backward()
    assert torch.allclose(T0.grad.flatten(), T0.xgrad[0].flatten())

    T1: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O1: Tensor = T1.sum(dim=(1,))
    backward(source=O1, order=2)
    assert T1.xgrad[0].shape == (O1.numel(), T1.numel())


def test_13() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = T0.prod()
    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())

    O0.backward()
    assert torch.allclose(T0.grad.flatten(), T0.xgrad[0].flatten())

    T1: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O1: Tensor = T1.prod(dim=1)
    backward(source=O1, order=2)
    assert T1.xgrad[0].shape == (O1.numel(), T1.numel())


def test_14() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = T0.view(size=(4, 1, 4))
    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_15() -> None:

    T: Tensor = torch.rand(size=(4, 6), requires_grad=True)
    O: Tensor = torch.nn.functional.elu(T)
    backward(source=O, order=2)
    assert T.xgrad[0].shape == (O.numel(), T.numel())


def test_16() -> None:

    T: Tensor = torch.rand(size=(4, 6), requires_grad=True)
    O: Tensor = torch.nn.functional.leaky_relu(T)
    backward(source=O, order=2)
    assert T.xgrad[0].shape == (O.numel(), T.numel())


def test_17() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    T1: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    cond: Tensor = T0 > 0.5
    O0: Tensor = torch.where(condition=cond, input=T0, other=T1)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())


def test_18() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    T1: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    cond: Tensor = T0 > 0.5
    O0: Tensor = torch.where(condition=cond, input=T0, other=T1)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())


def test_19() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.selu(T0)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_20() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    T1: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.prelu(T0, T1.sum(dim=1))

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())


def test_21() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.pow(T0, 3.5)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    T1: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.pow(T0, T1)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())


def test_22() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.square(T0)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_23() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.sqrt(T0)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_24() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.exp(T0)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_25() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.log(T0)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_26() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.sin(T0)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_27() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = torch.cos(T0)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_28() -> None:

    T0: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    T1: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    O0: Tensor = T0 / T1

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())


def test_29() -> None:

    T0: Tensor = torch.rand(size=(2, 4, 6), requires_grad=True)
    T1: Tensor = torch.rand(size=(2, 6, 8), requires_grad=True)

    O0: Tensor = torch.bmm(T0, T1)
    O0 = O0.sum()

    backward(source=O0, order=2)
    O0.backward()

    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())
    assert torch.allclose(T0.xgrad[0].flatten(), T0.grad.flatten())
    assert torch.allclose(T1.xgrad[0].flatten(), T1.grad.flatten())


def test_30() -> None:

    T0: Tensor = torch.rand(size=(4,), requires_grad=True)
    T1: Tensor = torch.rand(size=(4,), requires_grad=True)
    O0: Tensor = torch.dot(T0, T1)

    backward(source=O0, order=2)
    assert T0.xgrad[0].shape == (O0.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O0.numel(), T1.numel())
