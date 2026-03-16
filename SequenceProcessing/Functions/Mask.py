from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.FunctionNode import FunctionNode
from Math.Tensor import Tensor


class Mask(Function):
    """
    Applies a causal mask to a tensor.
    """

    def __init__(self):
        """
        Constructor for Mask.
        """
        pass

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Applies the mask to the tensor.

        Elements above the diagonal are replaced with negative infinity.

        :param tensor: Input tensor.
        :return: Masked tensor.
        """
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                if j > i:
                    values.append(float("-inf"))
                else:
                    values.append(tensor.getValue((i, j)))

        return Tensor(values, shape)

    def derivative(self, value: Tensor, backward: Tensor) -> Tensor:
        """
        Calculates the derivative of the mask operation.

        Since the Java version multiplies the backward tensor by a tensor
        filled with ones, the backward tensor is preserved unchanged.

        :param value: Current tensor value.
        :param backward: Backward gradient tensor.
        :return: Resulting gradient tensor.
        """
        values = []
        shape = value.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                values.append(1.0)

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