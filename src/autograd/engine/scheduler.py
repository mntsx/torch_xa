# python 3.12

# Internal dependencies
from src.autograd.engine.graph import Graph, Node


class Scheduler:

    def __init__(self, graph: Graph) -> None:
        self._graph: Graph = graph

    def _sorting_key(self, node: Node) -> int:
        return -len(node.prenodes)

    def backprop(self) -> None:
        frontier: list[Node] = self._graph.heads
        while len(frontier) > 0:
            frontier.sort(key=self._sorting_key)
            expand_node: Node = frontier[0]
            expand_node.propagate()
            frontier = self._graph.heads
        return None
