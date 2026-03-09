import unittest
import math
from Math.Tensor import Tensor
from SequenceProcessing.Functions.AdditionByConstant import AdditionByConstant
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.Inverse import Inverse
from SequenceProcessing.Functions.Mean import Mean
from SequenceProcessing.Functions.Mask import Mask


class TestAdditionByConstant(unittest.TestCase):

    def test_calculate(self):
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

        self.assertEqual(result.getData(), [0.5, 0.5, 0.5])


class TestMultiplyByConstant(unittest.TestCase):

    def test_calculate(self):
        t = Tensor([1.0, 2.0, 3.0], (3,))
        f = MultiplyByConstant(2.0)

        out = f.calculate(t)

        self.assertEqual(out.getData(), [2.0, 4.0, 6.0])
        self.assertEqual(out.getShape(), (3,))

    def test_derivative(self):
        t = Tensor([1.0, 2.0, 3.0], (3,))
        grad = Tensor([1.0, 1.0, 1.0], (3,))

        f = MultiplyByConstant(2.0)

        out = f.derivative(t, grad)

        self.assertEqual(out.getData(), [2.0, 2.0, 2.0])


class TestInverse(unittest.TestCase):

    def test_calculate(self):
        t = Tensor([2.0, 4.0, 5.0, 10.0], (2, 2))
        f = Inverse()

        out = f.calculate(t)

        self.assertEqual(out.getData(), [0.5, 0.25, 0.2, 0.1])
        self.assertEqual(out.getShape(), (2, 2))

    def test_derivative_literal_java(self):
        t = Tensor([2.0, 4.0, 5.0, 10.0], (2, 2))
        grad = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        f = Inverse()

        out = f.derivative(t, grad)

        self.assertEqual(out.getData(), [-4.0, -16.0, -25.0, -100.0])



class TestMean(unittest.TestCase):

    def test_calculate(self):
        tensor = Tensor([1.0, 3.0, 2.0, 4.0], (2, 2))
        func = Mean()

        result = func.calculate(tensor)

        self.assertEqual(result.getData(), [2.0, 2.0, 3.0, 3.0])
        self.assertEqual(result.getShape(), (2, 2))

    def test_derivative(self):
        tensor = Tensor([1.0, 3.0, 2.0, 4.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        func = Mean()

        result = func.derivative(tensor, backward)

        self.assertEqual(result.getData(), [0.5, 0.5, 0.5, 0.5])
        self.assertEqual(result.getShape(), (2, 2))



class TestMask(unittest.TestCase):

    def test_calculate(self):
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        func = Mask()

        result = func.calculate(tensor)
        data = result.getData()

        self.assertEqual(data[0], 1.0)
        self.assertTrue(math.isinf(data[1]) and data[1] < 0)
        self.assertEqual(data[2], 3.0)
        self.assertEqual(data[3], 4.0)
        self.assertEqual(result.getShape(), (2, 2))

    def test_derivative(self):
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        backward = Tensor([0.1, 0.2, 0.3, 0.4], (2, 2))
        func = Mask()

        result = func.derivative(tensor, backward)

        self.assertEqual(result.getData(), [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(result.getShape(), (2, 2))


if __name__ == "__main__":
    unittest.main()
