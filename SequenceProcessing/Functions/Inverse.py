from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Inverse(Function):
    """
    Computes the element-wise inverse of a tensor.
    """

    def __init__(self):
        """
        Constructor for Inverse.
        """
        pass

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Calculates the element-wise inverse of the input tensor.

        :param tensor: Input tensor.
        :return: Output tensor whose elements are 1 / x.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(1.0 / tensor.getValue((i, j)))

        return Tensor(values, shape)

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        """
        :param tensor: Input tensor.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_value = tensor.getValue((i, j))
                values.append(-(current_value ** 2))

        return backward.hadamardProduct(Tensor(values, shape))

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