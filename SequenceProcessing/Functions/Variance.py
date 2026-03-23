from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Variance(Function):
    """
    Computes row-wise variance-like values.
    """

    def __init__(self):
        """
        Constructor for Variance.
        """
        pass

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        :param tensor: Input tensor.
        :return: Result tensor.
        """
        values = []
        variances = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            total = 0.0
            for j in range(shape[1]):
                val = tensor.getValue((i, j))
                total += val ** 2
            variances.append(total / shape[1])

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(variances[i])

        return Tensor(values, shape)

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        """
        :param tensor: Input tensor.
        :param backward: Backward gradient tensor.
        :return: Result gradient tensor.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                val = tensor.getValue((i, j))
                values.append(2.0 * ((shape[1] * val) ** 0.5) / shape[1])

        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input nodes.
        :param is_biased: Bias flag.
        :return: New computational node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node