from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Optimizer.Optimizer import Optimizer


class TransformerParameter(NeuralNetworkParameter):
    """
    Parameter class for transformer-based neural networks.
    """

    __L: int
    __N: int
    __V: int
    __epsilon: float

    __input_hidden_layers: List[int]
    __output_hidden_layers: List[int]

    __input_functions: List[Function]
    __output_functions: List[Function]

    __gamma_input_values: List[float]
    __gamma_output_values: List[float]

    __beta_input_values: List[float]
    __beta_output_values: List[float]

    def __init__(self,
                 seed: int,
                 epoch: int,
                 optimizer: Optimizer,
                 initialization: Initialization,
                 loss: Function,
                 word_embedding_length: int,
                 multi_head_attention_length: int,
                 vocabulary_length: int,
                 epsilon: float,
                 input_hidden_layers: List[int],
                 output_hidden_layers: List[int],
                 input_activation_functions: List[Function],
                 output_activation_functions: List[Function],
                 gamma_input_values: List[float],
                 gamma_output_values: List[float],
                 beta_input_values: List[float],
                 beta_output_values: List[float]):
        """
        Constructor for TransformerParameter.

        :param seed: Random seed.
        :param epoch: Number of epochs.
        :param optimizer: Optimization algorithm.
        :param initialization: Initialization method.
        :param loss: Loss function.
        :param word_embedding_length: Embedding size.
        :param multi_head_attention_length: Number of attention heads.
        :param vocabulary_length: Vocabulary size.
        :param epsilon: Small constant for numerical stability.
        :param input_hidden_layers: Input hidden layer sizes.
        :param output_hidden_layers: Output hidden layer sizes.
        :param input_activation_functions: Input activation functions.
        :param output_activation_functions: Output activation functions.
        :param gamma_input_values: Gamma values (input normalization).
        :param gamma_output_values: Gamma values (output normalization).
        :param beta_input_values: Beta values (input normalization).
        :param beta_output_values: Beta values (output normalization).
        """
        super().__init__(seed, epoch, optimizer, initialization, loss, 0.0, 1)

        self.__L = word_embedding_length + 1
        self.__N = multi_head_attention_length
        self.__V = vocabulary_length
        self.__epsilon = epsilon

        self.__input_hidden_layers = input_hidden_layers
        self.__output_hidden_layers = output_hidden_layers

        self.__input_functions = input_activation_functions
        self.__output_functions = output_activation_functions

        self.__gamma_input_values = gamma_input_values
        self.__gamma_output_values = gamma_output_values

        self.__beta_input_values = beta_input_values
        self.__beta_output_values = beta_output_values

    def getGammaInputValue(self, index: int) -> float:
        """
        Returns gamma input value at index.
        """
        return self.__gamma_input_values[index]

    def getGammaOutputValue(self, index: int) -> float:
        """
        Returns gamma output value at index.
        """
        return self.__gamma_output_values[index]

    def getBetaInputValue(self, index: int) -> float:
        """
        Returns beta input value at index.
        """
        return self.__beta_input_values[index]

    def getBetaOutputValue(self, index: int) -> float:
        """
        Returns beta output value at index.
        """
        return self.__beta_output_values[index]

    def getEpsilon(self) -> float:
        """
        Returns epsilon value.
        """
        return self.__epsilon

    def getDk(self) -> int:
        """
        Returns dimension per attention head.
        """
        return self.__L // self.__N

    def getL(self) -> int:
        """
        Returns embedding dimension (L).
        """
        return self.__L

    def getN(self) -> int:
        """
        Returns number of attention heads.
        """
        return self.__N

    def getV(self) -> int:
        """
        Returns vocabulary size.
        """
        return self.__V

    def getInputHiddenLayer(self, index: int) -> int:
        """
        Returns input hidden layer size at index.
        """
        return self.__input_hidden_layers[index]

    def getOutputHiddenLayer(self, index: int) -> int:
        """
        Returns output hidden layer size at index.
        """
        return self.__output_hidden_layers[index]

    def getInputActivationFunction(self, index: int) -> Function:
        """
        Returns input activation function at index.
        """
        return self.__input_functions[index]

    def getOutputActivationFunction(self, index: int) -> Function:
        """
        Returns output activation function at index.
        """
        return self.__output_functions[index]

    def getInputSize(self) -> int:
        """
        Returns number of input layers.
        """
        return len(self.__input_hidden_layers)

    def getOutputSize(self) -> int:
        """
        Returns number of output layers.
        """
        return len(self.__output_hidden_layers)