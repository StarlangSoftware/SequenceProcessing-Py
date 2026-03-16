from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Variance(Function):
    """
    Computes the mean of squared values for each row and repeats it across the row.
    """

    def __init__(self):
        """
        Constructor for Variance.
        """
        pass

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Calculates the mean of squared values for each row and fills the row with that value.

        Example:
        [[1, 2, 3],
         [4, 5, 6]]

        becomes

        [[(1^2+2^2+3^2)/3, ...],
         [(4^2+5^2+6^2)/3, ...]]

        :param tensor: Input tensor.
        :return: Tensor where each row contains its mean squared value.
        """
        values = []
        variances = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            total = 0.0
            for j in range(shape[1]):
                current_value = tensor.getValue((i, j))
                total += current_value * current_value
            variances.append(total / shape[1])

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(variances[i])

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the mean-of-squares operation.

        If
            f(x) = (1 / n) * sum(x_i^2)
        then
            df/dx = 2x / n

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_value = value.getValue((i, j))
                values.append((2.0 * current_value) / shape[1])

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