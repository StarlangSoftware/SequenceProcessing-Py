from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class RemoveBias(Function):
    """
    Removes the last bias value from a tensor.
    """

    def __init__(self):
        """
        Constructor for RemoveBias.
        """
        pass

    def calculate(self, matrix: Tensor) -> Tensor:
        """
        Removes the last element of the tensor data.

        :param matrix: Input tensor.
        :return: Tensor without the last element.
        """
        data = matrix.getData()
        values = []

        for i in range(len(data) - 1):
            values.append(data[i])

        return Tensor(values, (1, len(values)))

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the remove bias operation.

        During backpropagation, a zero is appended for the removed bias term.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        values = backward.getData()
        new_values = list(values)
        new_values.append(0.0)

        return Tensor(new_values, (1, len(new_values)))

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