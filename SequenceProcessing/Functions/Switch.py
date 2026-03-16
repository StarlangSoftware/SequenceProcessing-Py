from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Switch(Function):
    """
    Switches the tensor output on or off depending on the turn value.
    """

    __turn: bool

    def __init__(self):
        """
        Constructor for Switch.
        """
        self.__turn = True

    def getTurn(self) -> bool:
        """
        Getter for turn.

        :return: Current turn value.
        """
        return self.__turn

    def setTurn(self, turn: bool) -> None:
        """
        Setter for turn.

        :param turn: New turn value.
        """
        self.__turn = turn

    def calculate(self, matrix: Tensor) -> Tensor:
        """
        Returns the input tensor if turn is True.
        Otherwise returns a zero tensor with the same shape.

        :param matrix: Input tensor.
        :return: Output tensor.
        """
        if self.__turn:
            return matrix

        values = []
        size = 1
        shape = matrix.getShape()

        for i in range(len(shape)):
            size *= shape[i]

        for i in range(size):
            values.append(0.0)

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the switch operation.

        If turn is True, the backward tensor is passed unchanged.
        Otherwise, a zero tensor with the same shape as value is returned.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        if self.__turn:
            return backward

        return self.calculate(value)

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