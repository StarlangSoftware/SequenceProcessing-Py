import unittest
from Math.Tensor import Tensor
from SequenceProcessing.Functions.AdditionByConstant import AdditionByConstant


class TestAdditionByConstant(unittest.TestCase):

    def test_calculate(self):
        # input tensor
        tensor = Tensor([1.0, 2.0, 3.0], [3])

        func = AdditionByConstant(2.0)

        result = func.calculate(tensor)

        self.assertEqual(result.getData(), [3.0, 4.0, 5.0])
        self.assertEqual(result.getShape(), [3])

    def test_derivative(self):
        tensor = Tensor([1.0, 2.0, 3.0], [3])
        grad = Tensor([0.5, 0.5, 0.5], [3])

        func = AdditionByConstant(2.0)

        result = func.derivative(tensor, grad)

        # derivative should return gradient unchanged
        self.assertEqual(result.getData(), [0.5, 0.5, 0.5])


if __name__ == "__main__":
    unittest.main()