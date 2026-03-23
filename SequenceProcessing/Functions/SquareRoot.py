from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class SquareRoot(Function):
    """
    Computes the element-wise square root of (epsilon + tensor).
    """

    __epsilon: float

    def __init__(self, epsilon: float):
        """
        Constructor for SquareRoot.

        :param epsilon: Small constant added before square root.
        """
        self.__epsilon = float(epsilon)

    def getEpsilon(self) -> float:
        """
        Getter for epsilon.

        :return: epsilon value.
        """
        return self.__epsilon

    def setEpsilon(self, epsilon: float) -> None:
        """
        Setter for epsilon.

        :param epsilon: New epsilon value.
        """
        self.__epsilon = float(epsilon)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Computes sqrt(epsilon + x) element-wise.

        :param tensor: Input tensor.
        :return: Result tensor.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                val = tensor.getValue((i, j))
                values.append((self.__epsilon + val) ** 0.5)

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
                values.append(1.0 / (2.0 * val))

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