from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Mean(Function):
    """
    Computes the mean of each row of a tensor and repeats the mean across the row.
    """

    def __init__(self):
        """
        Constructor for Mean.
        """
        pass

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Calculates the mean of each row and fills the row with that mean.

        :param tensor: Input tensor.
        :return: Tensor where each row contains its mean value.
        """
        values = []
        means = []
        shape = tensor.getShape()

        # compute row means
        for i in range(shape[0]):
            total = 0.0
            for j in range(shape[1]):
                total += tensor.getValue((i, j))
            means.append(total / shape[1])

        # fill rows with mean
        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(means[i])

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the mean operation.

        Each element contributes equally to the row mean,
        so the gradient is 1 / row_size.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(1.0 / shape[1])

        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                input_nodes: List[ComputationalNode],
                is_biased: bool) -> ComputationalNode:
        """
        Adds this function as an edge to the computational graph.

        :param input_nodes: Input computational nodes.
        :param is_biased: Indicates whether the connection is biased.
        :return: Newly created computational node.
        """
        new_node = FunctionNode(is_biased, self)
        input_nodes[0].add(new_node)
        return new_node