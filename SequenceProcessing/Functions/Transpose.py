from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Transpose(Function):
    """
    Transposes a 2D tensor.
    """

    def __init__(self):
        """
        Constructor for Transpose.
        """
        pass

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Performs the forward pass of the transpose operation.

        :param tensor: Input tensor.
        :return: Transposed tensor.
        """
        return tensor.transpose((1, 0))

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the transpose operation.

        The derivative of transpose is also transpose.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Transposed backward gradient tensor.
        """
        return backward.transpose((1, 0))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the edge is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node