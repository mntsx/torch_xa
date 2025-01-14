# python 3.12

# Standard Library dependencies
from enum import Enum
from typing import Optional, Tuple, Type, Union

# PyTorch dependencies
import torch
from torch import Tensor

# Internal dependencies
from src.autograd.configurations.selectors import Selector
from src.autograd.XAF.base import ExtendedAutogradFunction
from src.utils.partials import get_backward_idx, start_partials, sum_partials
from src.utils.types import AutogradFunction, MultiPartials, Partials, ShapedPartials


class S(Enum):
    L = 0  # locked
    U = 1  # unlocked
    R = 2  # available
    P = 3  # propagated


class Node:

    def __init__(self, grad_fn: AutogradFunction, idx: int) -> None:
        # identity related attributes
        self._grad_fn: AutogradFunction = grad_fn
        self._idx: int = idx

        # graph related attributes
        self._active: bool = False
        self._subnodes: set["Node"] = set()
        self._prenodes: set["Node"] = set()

        # computation related attributes
        self._status: int = S.L
        self._XAF: Union[None, Type[ExtendedAutogradFunction]] = None

        return None

    def assert_node_integrity(self) -> None:
        assert len(self._subnodes) + len(self._prenodes) > 0

    def register_edge(self, prenode: "Node", subnode: "Node") -> None:
        if prenode is self:
            self._subnodes.add(subnode)
        else:
            self._prenodes.add(prenode)
        return None

    def acquire_XAF(self, order: int, selector: Selector, device: torch.device) -> None:
        self._XAF = selector(grad_fn=self._grad_fn, order=order, device=device)
        return None

    def register_XAF_idx(self) -> None:
        assert self._XAF is not None
        self._XAF.register_idx(idx=self._idx)
        return None

    @property
    def grad_fn(self) -> Tensor:
        return self._grad_fn

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def active(self) -> bool:
        self.assert_node_integrity()
        return self._active

    @active.setter
    def active(self, value: bool) -> None:
        self.assert_node_integrity()
        self._active = value
        return None

    @property
    def partials(self) -> MultiPartials:
        partials: MultiPartials = self._XAF.partials
        return partials

    @property
    def XAF(self) -> Type[ExtendedAutogradFunction]:
        assert self._XAF is not None
        return self._XAF

    @XAF.setter
    def XAF(self, value: Type[ExtendedAutogradFunction]) -> None:
        self._XAF = value
        return None

    @property
    def is_leaf(self) -> bool:
        is_leaf: bool = len(self._subnodes) == 0
        return is_leaf

    @property
    def status(self) -> int:
        self.assert_node_integrity()
        return self._status

    @property
    def subnodes(self) -> set["Node"]:
        self.assert_node_integrity()
        return set(self._subnodes)

    @property
    def prenodes(self) -> set["Node"]:
        self.assert_node_integrity()
        return set(self._prenodes)

    def unlock(self) -> None:
        self.assert_node_integrity()
        if all([node.status == S.R for node in self._prenodes]) and self._active:
            self._status = S.U
        return None

    def close(self) -> None:
        self.assert_node_integrity()
        assert self._status in [S.R, S.L]
        if all([node.status in [S.R, S.L] for node in self._subnodes]):
            self._status = S.L
            equal_XAF_nodes: set["Node"] = set()
            for subnode in self._subnodes:
                for equinode in subnode.prenodes:
                    if equinode.XAF is self._XAF:
                        equal_XAF_nodes.add(equinode)
            if all([node.status == S.L for node in equal_XAF_nodes]):
                self._XAF.clear_partials()
        return None

    def _get_preXAF_posidx(self, preXAF: Type[ExtendedAutogradFunction]) -> int:
        idx: Union[None, int] = None
        for i, (next_function, _) in enumerate(preXAF.grad_fn.next_functions):
            if next_function is self._grad_fn:
                idx = i
        assert isinstance(idx, int)
        return idx

    def propagate(self) -> None:
        self.assert_node_integrity()
        assert self._status == S.U
        assert all([node.status == S.R for node in self._prenodes])
        # gather all partials from parent nodes
        # Note
        # child nodes always share grad_fn
        # parent nodes can either share it or not
        preXAFs: set[Type[ExtendedAutogradFunction]]
        preXAFs = set([node.XAF for node in self._prenodes])
        prepartials_list: list[ShapedPartials] = list()
        for preXAF in sorted(preXAFs, key=lambda fn: str(fn)):  # type(fn).__name__
            preXAF_posidx: int = self._get_preXAF_posidx(preXAF=preXAF)
            prepartials_list.append(preXAF.partials[preXAF_posidx])
        prepartials: ShapedPartials = sum_partials(partials_list=prepartials_list)
        # gc already collected partials
        for node in self._prenodes:
            node.close()
        # call XAF
        self._XAF(shaped_output_partials=prepartials, idx=self._idx)
        # unlock subnodes
        self._status = S.R
        for node in self._subnodes:
            node.unlock()
        return None


class SourceNode(Node):

    def __init__(self, grad_fn: AutogradFunction, idx: int, tensor: Tensor) -> None:
        # super().__init__(self, grad_fn=grad_fn, idx=idx)
        super().__init__(grad_fn=grad_fn, idx=idx)  # Fixed: removed self parameter
        self._tensor: Tensor = tensor
        self._status: int = S.U
        self._order: Union[None, int] = None
        return None

    @property
    def order(self) -> int:
        return self._order

    @order.setter
    def order(self, value: int) -> None:
        self._order = value
        return None

    def propagate(self) -> None:
        self.assert_node_integrity()
        assert self._status == S.U
        assert self._order is not None
        prepartials: ShapedPartials
        prepartials = start_partials(
            tensor=self._tensor, order=self._order, device=self._tensor.device
        )
        for node in self._prenodes:
            node.close()
        self._XAF(shaped_output_partials=prepartials, idx=self._idx)
        self._status = S.R
        for node in self._subnodes:
            node.unlock()
        return None


class Graph:

    # 1. Backprop path (creating new attributes in required tensors)
    # 2. if target is not None
    #       -> ignore unnecesary paths from target
    #    else:
    #       -> ignore unnecesary paths
    # 3. Backprop path (assigning partials_fn(s) and indicating expected calls)

    def __init__(self) -> None:
        self._source_node: SourceNode
        self._device: torch.device
        self._nodes: dict[Tuple[AutogradFunction, int], Type[Node]] = dict()
        self._XAFs: dict[AutogradFunction, Type[ExtendedAutogradFunction]] = dict()

    @property
    def heads(self) -> list[Node]:
        heads: list[Node] = [
            node for node in self._nodes.values() if node.status == S.U
        ]
        return heads

    @property
    def leafs(self) -> list[Node]:
        leafs: list[Node] = [
            node for node in self._nodes.values() if len(node.subnodes) == 0
        ]
        return leafs

    @classmethod
    def construct(cls, source: Tensor) -> "Graph":
        graph: "Graph" = cls()
        graph._device = source.device
        grad_fn: AutogradFunction = source.grad_fn
        idx: int = get_backward_idx(tensor=source)
        node: SourceNode = SourceNode(grad_fn=grad_fn, idx=idx, tensor=source)
        graph._source_node = node
        graph._nodes[(grad_fn, idx)] = node
        graph._retrieve_subnodes(grad_fn=grad_fn, idx=idx)
        return graph

    def _retrieve_subnodes(self, grad_fn: AutogradFunction, idx: int) -> None:
        node: Node = self._nodes[(grad_fn, idx)]
        for next_fn, next_idx in grad_fn.next_functions:
            if next_fn is not None:
                next_node: Node
                if (next_fn, next_idx) in self._nodes.keys():
                    next_node = self._nodes[(next_fn, next_idx)]
                else:
                    next_node = Node(grad_fn=next_fn, idx=next_idx)
                    self._nodes[(next_fn, next_idx)] = next_node
                node.register_edge(prenode=node, subnode=next_node)
                next_node.register_edge(prenode=node, subnode=next_node)
                self._retrieve_subnodes(grad_fn=next_fn, idx=next_idx)

        return None

    def load_XAFs(self, order: int, selector: Selector) -> None:
        self._nodes.clear()
        node: SourceNode = self._source_node
        node.order = order
        self._recursive_XAF_load(order=order, selector=selector, node=node)

    def _recursive_XAF_load(self, order: int, selector: Selector, node: Node) -> None:
        idx: int = node.idx
        grad_fn: AutogradFunction = node.grad_fn

        if (grad_fn, idx) not in self._nodes:
            self._nodes[(grad_fn, idx)] = node
            if grad_fn not in self._XAFs:
                node.acquire_XAF(order=order, selector=selector, device=self._device)
                node.register_XAF_idx()
                self._XAFs[grad_fn] = node.XAF
            else:
                node.XAF = self._XAFs[grad_fn]
                node.register_XAF_idx()

        for next_node in node.subnodes:
            self._recursive_XAF_load(order=order, selector=selector, node=next_node)

    def clear_partials(self) -> None:
        for node in self.leafs:
            if node.active and node.is_leaf and "variable" in dir(node.grad_fn):
                if "xgrad" in dir(node.grad_fn.variable):
                    node.close()
        return None

    def attach_partials(self) -> None:
        for node in self.leafs:
            if node.is_leaf and "variable" in dir(node.grad_fn):
                partials: Union[None, Partials] = None
                if node.partials is not None:
                    partials = node.partials[0][0]
                setattr(node.grad_fn.variable, "xgrad", partials)
        return None

    def remove_partials(self) -> None:
        for node in self.leafs:
            if node.is_leaf and "variable" in dir(node.grad_fn):
                if "xgrad" in dir(node.grad_fn.variable):
                    delattr(node.grad_fn.variable, "xgrad")
        return None

    def prune(self, target: Optional[Tensor] = None) -> None:
        for node in self._nodes.values():
            node.active = False
        if target is None:
            self._standard_pruning(node=self._source_node)
        else:
            assert isinstance(target, Tensor), f"target must be a {Tensor.__name__}."
            target_node: Union[None, Node] = None
            for node in self.leafs:
                fn: AutogradFunction = node.grad_fn
                if "variable" in dir(fn) and fn.variable is target:
                    target_node = node
            assert target_node is not None, "target not found among graph leafs"
            target_node.active = self._target_pruning(node=target_node)

        return None

    def _standard_pruning(self, node: Node) -> None:
        is_target: bool = node.is_leaf
        connects_to_target: bool = False
        for next_node in node.subnodes:
            next_connected: bool = self._standard_pruning(node=next_node)
            connects_to_target = connects_to_target or next_connected
        connects_to_target = is_target or connects_to_target
        node.active = connects_to_target
        return connects_to_target

    def _target_pruning(self, node: Node) -> bool:
        is_source: bool = node is self._source_node
        connects_to_source: bool = False
        for next_node in node.prenodes:
            next_connected: bool = self._target_pruning(node=next_node)
            connects_to_source = connects_to_source or next_connected
        connects_to_source = is_source or connects_to_source
        node.active = connects_to_source
        return connects_to_source
