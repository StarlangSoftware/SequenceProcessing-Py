import unittest
import math
from Math.Tensor import Tensor
from SequenceProcessing.Functions.AdditionByConstant import AdditionByConstant
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.Inverse import Inverse
from SequenceProcessing.Functions.Mean import Mean
from SequenceProcessing.Functions.Mask import Mask
from SequenceProcessing.Functions.RemoveBias import RemoveBias
from SequenceProcessing.Functions.SquareRoot import SquareRoot
from SequenceProcessing.Functions.Switch import Switch
from SequenceProcessing.Functions.Transpose import Transpose
from SequenceProcessing.Functions.Variance import Variance


class VarianceTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests forward computation.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        func = Variance()

        result = func.calculate(tensor)

        # Row-wise mean of squares:
        # row1: (1^2 + 2^2)/2 = (1 + 4)/2 = 2.5
        # row2: (3^2 + 4^2)/2 = (9 + 16)/2 = 12.5
        expected = [2.5, 2.5, 12.5, 12.5]

        self.assertEqual(expected, result.getData())
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        """
        Tests derivative.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        func = Variance()

        result = func.derivative(tensor, backward)

        shape = tensor.getShape()
        expected = []

        for val in tensor.getData():
            expected.append(2.0 * ((shape[1] * val) ** 0.5) / shape[1])

        self.assertEqual(expected, result.getData())
        self.assertEqual((2, 2), result.getShape())


class TransposeTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests the forward transpose operation.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        function = Transpose()

        result = function.calculate(tensor)

        self.assertEqual([1.0, 3.0, 2.0, 4.0], result.getData())
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        """
        Tests the backward transpose operation.
        """
        value = Tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        backward = Tensor([1.0, 3.0, 2.0, 4.0], (2, 2))

        function = Transpose()

        result = function.derivative(value, backward)

        self.assertEqual([1.0, 2.0, 3.0, 4.0], result.getData())
        self.assertEqual((2, 2), result.getShape())


class SwitchTest(unittest.TestCase):

    def testCalculateWhenTurnIsTrue(self):
        """
        Tests calculate when turn is True.
        """
        tensor = Tensor([1.0, 2.0, 3.0], (1, 3))
        function = Switch()

        result = function.calculate(tensor)

        self.assertEqual([1.0, 2.0, 3.0], result.getData())
        self.assertEqual((1, 3), result.getShape())

    def testCalculateWhenTurnIsFalse(self):
        """
        Tests calculate when turn is False.
        """
        tensor = Tensor([1.0, 2.0, 3.0], (1, 3))
        function = Switch()
        function.setTurn(False)

        result = function.calculate(tensor)

        self.assertEqual([0.0, 0.0, 0.0], result.getData())
        self.assertEqual((1, 3), result.getShape())

    def testDerivativeWhenTurnIsTrue(self):
        """
        Tests derivative when turn is True.
        """
        value = Tensor([1.0, 2.0, 3.0], (1, 3))
        backward = Tensor([0.5, 0.6, 0.7], (1, 3))
        function = Switch()

        result = function.derivative(value, backward)

        self.assertEqual([0.5, 0.6, 0.7], result.getData())
        self.assertEqual((1, 3), result.getShape())

    def testDerivativeWhenTurnIsFalse(self):
        """
        Tests derivative when turn is False.
        """
        value = Tensor([1.0, 2.0, 3.0], (1, 3))
        backward = Tensor([0.5, 0.6, 0.7], (1, 3))
        function = Switch()
        function.setTurn(False)

        result = function.derivative(value, backward)

        self.assertEqual([0.0, 0.0, 0.0], result.getData())
        self.assertEqual((1, 3), result.getShape())


class SquareRootTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests forward computation.
        """
        tensor = Tensor([3.0, 8.0, 15.0, 24.0], (2, 2))
        func = SquareRoot(1.0)

        result = func.calculate(tensor)

        # sqrt(1 + x)
        expected = [
            (1 + 3.0) ** 0.5,
            (1 + 8.0) ** 0.5,
            (1 + 15.0) ** 0.5,
            (1 + 24.0) ** 0.5
        ]

        self.assertEqual(expected, result.getData())
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        tensor = Tensor([2.0, 4.0, 5.0, 10.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        func = SquareRoot(1.0)

        result = func.derivative(tensor, backward)

        expected = [
            1.0 / (2 * 2.0),
            1.0 / (2 * 4.0),
            1.0 / (2 * 5.0),
            1.0 / (2 * 10.0)
        ]

        self.assertEqual(expected, result.getData())
        self.assertEqual((2, 2), result.getShape())



class RemoveBiasTest(unittest.TestCase):

    def testCalculate(self):
        """
        Tests the forward computation of RemoveBias.
        """
        tensor = Tensor([1.0, 2.0, 3.0, 99.0], (1, 4))
        function = RemoveBias()

        result = function.calculate(tensor)

        self.assertEqual([1.0, 2.0, 3.0], result.getData())
        self.assertEqual((1, 3), result.getShape())

    def testDerivative(self):
        """
        Tests the backward computation of RemoveBias.
        """
        value = Tensor([1.0, 2.0, 3.0], (1, 3))
        backward = Tensor([0.5, 0.6, 0.7], (1, 3))
        function = RemoveBias()

        result = function.derivative(value, backward)

        self.assertEqual([0.5, 0.6, 0.7, 0.0], result.getData())
        self.assertEqual((1, 4), result.getShape())



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
        t = Tensor([1.0, 2.0, 3.0], (1, 3))
        f = MultiplyByConstant(2.0)

        out = f.calculate(t)

        self.assertEqual(out.getData(), [2.0, 4.0, 6.0])
        self.assertEqual(out.getShape(), (1, 3))

    def test_derivative(self):
        t = Tensor([1.0, 2.0, 3.0], (1, 3))
        grad = Tensor([1.0, 1.0, 1.0], (1, 3))

        f = MultiplyByConstant(2.0)

        out = f.derivative(t, grad)

        self.assertEqual(out.getData(), [2.0, 2.0, 2.0])


class InverseTest(unittest.TestCase):

    def testCalculate(self):
        tensor = Tensor([2.0, 4.0, 5.0, 10.0], (2, 2))
        function = Inverse()

        result = function.calculate(tensor)

        self.assertEqual([0.5, 0.25, 0.2, 0.1], result.getData())
        self.assertEqual((2, 2), result.getShape())

    def testDerivative(self):
        tensor = Tensor([2.0, 4.0, 5.0, 10.0], (2, 2))
        backward = Tensor([1.0, 1.0, 1.0, 1.0], (2, 2))
        function = Inverse()

        result = function.derivative(tensor, backward)

        self.assertEqual([-4.0, -16.0, -25.0, -100.0], result.getData())
        self.assertEqual((2, 2), result.getShape())




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
