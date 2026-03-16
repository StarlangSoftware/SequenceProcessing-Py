from ComputationalGraph.Function import Function
from Math.Tensor import Tensor


class AdditionByConstant(Function):
    constant: float

    def __init__(self, constant: float):
        """
        Initializes the AdditionByConstant function.

        :param constant: Constant value to be added to each tensor element.
        """
        self.constant = float(constant)

    def calculate(self, tensor: Tensor) -> Tensor:
        """
        Adds the constant value to each element of the tensor.

        :param tensor: Input tensor.
        :return: Tensor after adding the constant to each value.
        """
        values = [v + self.constant for v in tensor.getData()]
        return Tensor(values, tensor.getShape())

    def derivative(self, tensor: Tensor, tensor1: Tensor) -> Tensor:
        """
        Computes the derivative of the addition operation.

        :param tensor: Input tensor.
        :param tensor1: Gradient tensor.
        :return: The propagated gradient tensor.
        """
        return tensor1

    def getConstant(self) -> float:
        """
        Returns the constant value.

        :return: constant
        """
        return self.constant