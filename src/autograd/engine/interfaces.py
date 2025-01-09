# python 3.12

# Standard Library dependencies
from typing import Iterable, Optional, Type

# PyTorch dependencies
from torch import Tensor

# Internal dependencies
from src.autograd.engine.graph import Graph
from src.autograd.engine.scheduler import Scheduler
from src.autograd.configurations.exchangers import Selector, XAFexchanger
from src.autograd.configurations.selectors import XAFselector, DefaultSelector
from src.utils.types import AutogradFunction


class Superset:

    def __init__(self, graph: Graph) -> None:
        self._graph: Graph = graph

    @classmethod
    def construct(cls, source: Tensor, batch: Optional[bool] = True) -> "Superset":
        graph: Graph = Graph.construct(source=source, batch=batch)
        superset: "Superset" = cls(graph=graph)
        return superset

    def clear_partials(self) -> None:
        self._graph.clear_partials()
        self._graph.attach_partials()
        return None

    def attach_partials(self) -> None:
        self._graph.attach_partials()
        return None

    def remove_partials(self) -> None:
        self._graph.remove_partials()
        return None

    def backward(
        self,
        order: int,
        target: Optional[Tensor] = None,
        configuration: list[callable] = [],
    ) -> None:
        raise NotImplementedError()
        return None

    def operator_partials(self, tensor: Tensor) -> None:
        # get partials of tensor's associated operator XAF
        raise NotImplementedError()

    def retain_partials(self, tensors: Iterable[Tensor]) -> None:
        fns: set[AutogradFunction] = set([T.grad_fn for T in tensors])
        for node in self._graph._nodes.values():
            if node.grad_fn in fns:
                node.XAF.retain_partials = True
        return None

    def clear_retentions(self, tensors: Optional[Iterable[Tensor]] = None) -> None:
        if tensors is None:
            for node in self._graph._nodes.values():
                node.XAF.retain_partials = False
        else:
            fns: set[AutogradFunction] = set([T.grad_fn for T in tensors])
            for node in self._graph._nodes.values():
                if node.grad_fn not in fns:
                    node.XAF.retain_partials = False
        return None


def backward(
    source: Tensor,
    order: int,
    target: Optional[Tensor] = None,
    configurations: list[object] = [],
) -> Superset:

    assert isinstance(source, Tensor), f"source must be a {Tensor.__name__}."
    assert source.requires_grad, "source tensor must satisfy requires_grad=True."
    assert isinstance(order, int), "order must be a possitive integer."
    assert order >= 0, "order must be a possitive integer."
    if target is not None:
        assert isinstance(target, Tensor), f"target must be a {Tensor.__name__}."
        assert target.requires_grad, "target tensor must satisfy requires_grad=True."

    # validate and apply XAF selection configurations
    XAF_selector: Type[XAFselector] = DefaultSelector
    XAF_exchangers: list[Type[XAFexchanger]]
    XAF_exchangers = [cfg for cfg in configurations if issubclass(cfg, XAFexchanger)]
    # check that all configurations are classes, not instances
    for cfg in configurations:
        assert isinstance(cfg, Type)
    # switch XAF selector if one is provided
    selector_provided: bool = False
    for cfg in configurations:
        if issubclass(cfg, XAFselector):
            if selector_provided:
                raise ValueError("got more than one XAFselector.")
            else:
                selector_provided = True
                XAF_selector = cfg
    # apply additional XAF selector exchangers
    for exchangerA in XAF_exchangers:
        for exchangerB in XAF_exchangers:
            if exchangerA is not exchangerB:
                if len(set(exchangerA.targets) & set(exchangerB.targets)) > 0:
                    raise ValueError("collision between XAFexchangers targets.")
                cfg.set_selector(XAF_selector)
                XAF_selector = cfg.get_selector()

    # get the selector function
    selector_fn: Selector = XAF_selector._select

    # construct the computational graph
    graph: Graph = Graph.construct(source=source)
    graph.load_XAFs(order=order, selector=selector_fn)
    graph.prune(target=target)

    # execute backprop
    scheduler: Scheduler = Scheduler(graph=graph)
    scheduler.backprop()

    # prepare return
    graph.attach_partials()
    superset: Superset = Superset(graph=graph)

    return superset
