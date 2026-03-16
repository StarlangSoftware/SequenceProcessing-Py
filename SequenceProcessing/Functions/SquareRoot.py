import math
from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class SquareRoot(Function):
    """
    Computes the square root of epsilon + x for each tensor element.
    """

    __epsilon: float

    def __init__(self, epsilon: float):
        """
        Constructor for SquareRoot.

        :param epsilon: Small constant added before square root.
        """
        self.__epsilon = epsilon

    def getEpsilon(self) -> float:
        """
        Getter for epsilon.

        :return: Epsilon value.
        """
        return self.__epsilon

    def setEpsilon(self, epsilon: float) -> None:
        """
        Setter for epsilon.

        :param epsilon: New epsilon value.
        """
        self.__epsilon = epsilon

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Calculates sqrt(epsilon + x) for each element.

        :param tensor: Input tensor.
        :return: Output tensor.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(math.sqrt(self.__epsilon + tensor.getValue((i, j))))

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of sqrt(epsilon + x).

        If
            f(x) = sqrt(epsilon + x)
        then
            f'(x) = 1 / (2 * sqrt(epsilon + x))

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                current_value = value.getValue((i, j))
                derivative_value = 1.0 / (2.0 * math.sqrt(self.__epsilon + current_value))
                values.append(derivative_value)

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