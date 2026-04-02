from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class MultiplyByConstant(Function):
    """
    Multiplies every element of a tensor with a constant value.
    """

    __constant: float

    def __init__(self, constant: float):
        """
        Constructor for MultiplyByConstant.

        :param constant: The constant value to multiply with tensor elements.
        """
        self.__constant = float(constant)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Performs the forward computation.

        Multiplies each element of the tensor by the constant.

        :param tensor: Input tensor.
        :return: Result tensor.
        """
        values = []
        tensor_values = tensor.getData()

        for val in tensor_values:
            values.append(self.__constant * val)

        return Tensor(values, tensor.getShape())

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Computes the derivative of the multiplication operation.

        If
            f(x) = c * x
        then
            f'(x) = c

        The gradient tensor is therefore filled with the constant.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(self.__constant)

        constant_tensor = Tensor(values, shape)

        return backward.hadamardProduct(constant_tensor)

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