from ComputationalGraph.Function import Function
from Math.Tensor import Tensor


class Mask(Function):

    def calculate(self, tensor: Tensor) -> Tensor:
        values = []
        shape = tensor.getShape()

        for i in range(shape[0]):
            for j in range(shape[1]):
                if j > i:
                    values.append(float("-inf"))
                else:
                    values.append(tensor.getValue((i, j)))

        return Tensor(values, shape)

    def derivative(self, tensor: Tensor, backward: Tensor) -> Tensor:
        values = [1.0] * len(tensor.getData())
        return backward.hadamardProduct(Tensor(values, tensor.getShape()))