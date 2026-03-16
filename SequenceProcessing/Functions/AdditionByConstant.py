from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class AdditionByConstant(Function):
    """
    Adds a constant value to each element of the input tensor.
    """

    __constant: float

    def __init__(self, constant: float):
        """
        Constructor for AdditionByConstant.

        :param constant: Constant value to add to each tensor element.
        """
        self.__constant = float(constant)

    def getConstant(self) -> float:
        """
        Getter for constant value.

        :return: Constant value.
        """
        return self.__constant

    def setConstant(self, constant: float) -> None:
        """
        Setter for constant value.

        :param constant: New constant value.
        """
        self.__constant = float(constant)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Adds the constant value to each element of the tensor.

        :param tensor: Input tensor.
        :return: Output tensor after addition.
        """
        values = []
        tensor_values = tensor.getData()

        for val in tensor_values:
            values.append(val + self.__constant)

        return Tensor(values, tensor.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the addition operation.

        Since derivative of x + c is 1, the gradient passes through unchanged.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Gradient tensor.
        """
        return backward

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