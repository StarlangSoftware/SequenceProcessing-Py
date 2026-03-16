from typing import List

from ComputationalGraph.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Inverse(Function):

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Calculates the element-wise inverse of the given tensor.

        :param tensor: Input tensor.
        :return: Tensor whose elements are the inverses of the input tensor elements.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(1.0 / tensor.getValue((i, j)))

        return Tensor(values, shape)

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        """
        Calculates the derivative of the inverse function and multiplies it
        element-wise with the backward tensor.
        
        :param tensor: Input tensor used in the forward pass.
        :param backward: Backward gradient tensor.
        :return: Result of the hadamard product of backward tensor and local derivative tensor.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                value = tensor.getValue((i, j))
                values.append(-1.0 / (value * value))

        return backward.hadamardProduct(Tensor(values, shape))

    def addEdge(self,
                inputNodes: List[ComputationalNode],
                isBiased: bool) -> ComputationalNode:
        """
        Adds this function as an edge in the computational graph.

        :param inputNodes: List of input computational nodes.
        :param isBiased: Bias information of the new node.
        :return: Newly created function node.
        """
        new_node = FunctionNode(isBiased, self)
        inputNodes[0].add(new_node)
        return new_node