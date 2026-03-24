import unittest

from Math.Tensor import Tensor

from ComputationalGraph.Function.CrossEntropyLoss import CrossEntropyLoss
from ComputationalGraph.Function.Sigmoid import Sigmoid
from ComputationalGraph.Function.Tanh import Tanh
from ComputationalGraph.Initialization.RandomInitialization import RandomInitialization
from ComputationalGraph.Optimizer.AdamW import AdamW

from Dictionary.VectorizedDictionary import VectorizedDictionary

from SequenceProcessing.Classification.Transformer import Transformer
from SequenceProcessing.Parameters.TransformerParameter import TransformerParameter

class DummyWordComparator:
    """
    Dummy comparator implementation.
    """

    def compare(self, word, word1) -> int:
        """
        Compares two words.

        :param word: First word.
        :param word1: Second word.
        :return: Comparison result.
        """
        return 0
    
class DummyWordComparator(DummyWordComparator):
    """
    Dummy comparator implementation.
    """

    def compare(self, word, word1) -> int:
        return 0


class TransformerTest(unittest.TestCase):

    def testInitialization(self):
        """
        Tests transformer initialization and training.
        """

        tensors = [
            Tensor(
                [
                    0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9,
                    0.35, 0.12, 0.27, 0.17, 0.41,
                    float("inf"),
                    0.27, 0.67, 0.41, 1,
                    0.37, 0.17, 0.41, 6,
                    0.17, 0.65, 0.87, 5,
                    0.97, 0.19, 0.51, 4
                ],
                (29,)
            ),

            Tensor(
                [
                    0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9,
                    0.35, 0.12, 0.27, 0.17, 0.41,
                    float("inf"),
                    0.27, 0.67, 0.41, 1,
                    0.37, 0.17, 0.41, 6,
                    0.77, 0.61, 0.27, 2
                ],
                (25,)
            ),

            Tensor(
                [
                    0.2, 0.7, 0.1, 0.3, 0.4, 0.8, 0.9,
                    0.35, 0.12, 0.27, 0.17, 0.41,
                    float("inf"),
                    1.2, 3.6, 7.1, 3,
                    5.4, 0.17, 9.8, 4,
                    0.77, 0.61, 0.27, 2
                ],
                (25,)
            )
        ]

        input_layers = [30, 15]

        input_functions = [Tanh(), Sigmoid()]
        output_functions = [Sigmoid(), Tanh()]

        gamma_input = [1.0, 1.0]
        gamma_output = [1.0, 1.0, 1.0]

        beta_input = [0.0, 0.0]
        beta_output = [0.0, 0.0, 0.0]

        parameter = TransformerParameter(
            seed=1,
            epoch=150,
            optimizer=AdamW(0.025, 0.99, 0.99, 0.999, 1e-10, 0.1),
            initialization=RandomInitialization(),
            loss=CrossEntropyLoss(),
            word_embedding_length=3,
            multi_head_attention_length=2,
            vocabulary_length=7,
            epsilon=1e-9,
            input_hidden_layers=input_layers,
            output_hidden_layers=input_layers,
            input_activation_functions=input_functions,
            output_activation_functions=output_functions,
            gamma_input_values=gamma_input,
            gamma_output_values=gamma_output,
            beta_input_values=beta_input,
            beta_output_values=beta_output
        )

        dictionary = VectorizedDictionary(DummyWordComparator())

        transformer = Transformer(parameter, dictionary)

        transformer.train(tensors)


if __name__ == "__main__":
    unittest.main()