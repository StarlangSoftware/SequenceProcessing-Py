import unittest
from SequenceProcessing.Parameters.TransformerParameter import TransformerParameter
from SequenceProcessing.Parameters.RecurrentNeuralNetworkParameter import RecurrentNeuralNetworkParameter


class DummyFunction:
    pass


class DummyOptimizer:
    pass


class DummyInitialization:
    pass


class RecurrentNeuralNetworkParameterTest(unittest.TestCase):

    def testBasicProperties(self):
        """
        Tests basic parameter properties.
        """
        hidden_layers = [64, 32]
        functions = [DummyFunction(), DummyFunction()]

        param = RecurrentNeuralNetworkParameter(
            seed=1,
            epoch=10,
            optimizer=DummyOptimizer(),
            initialization=DummyInitialization(),
            loss=DummyFunction(),
            hidden_layers=hidden_layers,
            functions=functions,
            class_label_size=5
        )

        self.assertEqual(2, param.size())
        self.assertEqual(5, param.getClassLabelSize())
        self.assertEqual(64, param.getHiddenLayer(0))
        self.assertEqual(32, param.getHiddenLayer(1))
        self.assertEqual(functions[0], param.getActivationFunction(0))


class DummyFunction:
    pass


class DummyOptimizer:
    pass


class DummyInitialization:
    pass


class TransformerParameterTest(unittest.TestCase):

    def testBasicProperties(self):
        """
        Tests transformer parameter behavior.
        """
        param = TransformerParameter(
            seed=1,
            epoch=10,
            optimizer=DummyOptimizer(),
            initialization=DummyInitialization(),
            loss=DummyFunction(),
            word_embedding_length=128,
            multi_head_attention_length=8,
            vocabulary_length=1000,
            epsilon=1e-6,
            input_hidden_layers=[64, 32],
            output_hidden_layers=[32, 16],
            input_activation_functions=[DummyFunction(), DummyFunction()],
            output_activation_functions=[DummyFunction(), DummyFunction()],
            gamma_input_values=[1.0, 1.0],
            gamma_output_values=[1.0, 1.0],
            beta_input_values=[0.0, 0.0],
            beta_output_values=[0.0, 0.0]
        )

        self.assertEqual(129, param.getL())
        self.assertEqual(8, param.getN())
        self.assertEqual(1000, param.getV())
        self.assertEqual(129 // 8, param.getDk())
        self.assertEqual(1e-6, param.getEpsilon())

        self.assertEqual(64, param.getInputHiddenLayer(0))
        self.assertEqual(32, param.getOutputHiddenLayer(0))

        self.assertEqual(2, param.getInputSize())
        self.assertEqual(2, param.getOutputSize())


if __name__ == "__main__":
    unittest.main()