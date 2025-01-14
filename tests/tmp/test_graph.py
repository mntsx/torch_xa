import torch
from torch import Tensor

# Internal dependencies
from src.autograd.engine.interfaces import backward, Superset
from src.autograd.configurations.selectors import TestSelector


def test_1() -> None:

    T0: Tensor = torch.tensor([0])

    a: Tensor = torch.rand(size=(4, 4), requires_grad=True)
    b: Tensor = torch.rand(size=(4, 4), requires_grad=True)

    c: Tensor = a @ b
    d: Tensor = torch.sin(c)

    # test no targeted backward
    superset: Superset
    superset = backward(source=d, order=2, configurations=[TestSelector])
    assert a.xgrad == (T0, T0)
    assert b.xgrad == (T0, T0)
    superset.clear_partials()
    assert a.xgrad is None
    assert b.xgrad is None
    superset.remove_partials()
    assert "xgrad" not in dir(a)
    assert "xgrad" not in dir(b)

    # test targeted backward
    superset = backward(source=d, order=2, target=a, configurations=[TestSelector])
    assert a.xgrad == (T0, T0)
    assert b.xgrad is None


def test_2() -> None:

    T0: Tensor = torch.rand(size=(8,), requires_grad=True)
    T1: Tensor = torch.rand(size=(4, 6), requires_grad=True)
    T2: Tensor = torch.rand(size=(6, 8), requires_grad=True)

    O0: Tensor = torch.addmm(input=T0, mat1=T1, mat2=T2)
    O1: Tensor = torch.relu(O0)
    O2: Tensor = torch.add(O1, T0)
    O3: Tensor = torch.softmax(O2, dim=1)
    backward(source=O3, order=3)

    assert T0.xgrad[0].shape == (O3.numel(), T0.numel())
    assert T1.xgrad[0].shape == (O3.numel(), T1.numel())
    assert T2.xgrad[0].shape == (O3.numel(), T2.numel())
