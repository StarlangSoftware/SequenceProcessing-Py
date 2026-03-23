from typing import List

from ComputationalGraph.Function.Function import Function
from ComputationalGraph.Initialization.Initialization import Initialization
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Optimizer.Optimizer import Optimizer


class RecurrentNeuralNetworkParameter(NeuralNetworkParameter):
    """
    Parameter class for recurrent neural networks.
    """

    __hidden_layers: List[int]
    __functions: List[Function]
    __class_label_size: int

    def __init__(self,
                 seed: int,
                 epoch: int,
                 optimizer: Optimizer,
                 initialization: Initialization,
                 loss: Function,
                 hidden_layers: List[int],
                 functions: List[Function],
                 class_label_size: int):
        """
        Constructor for RecurrentNeuralNetworkParameter.

        :param seed: Random seed.
        :param epoch: Number of epochs.
        :param optimizer: Optimization algorithm.
        :param initialization: Weight initialization method.
        :param loss: Loss function.
        :param hidden_layers: List of hidden layer sizes.
        :param functions: Activation functions for each layer.
        :param class_label_size: Number of output classes.
        """
        super().__init__(seed, epoch, optimizer, initialization, loss, 0.0, 1)

        self.__hidden_layers = hidden_layers
        self.__functions = functions
        self.__class_label_size = class_label_size

    def size(self) -> int:
        """
        Returns the number of hidden layers.

        :return: Number of hidden layers.
        """
        return len(self.__hidden_layers)

    def getClassLabelSize(self) -> int:
        """
        Getter for class label size.

        :return: Class label size.
        """
        return self.__class_label_size

    def getActivationFunction(self, index: int) -> Function:
        """
        Returns the activation function at the given index.

        :param index: Index of the function.
        :return: Activation function.
        """
        return self.__functions[index]

    def getHiddenLayer(self, index: int) -> int:
        """
        Returns the hidden layer size at the given index.

        :param index: Index of the hidden layer.
        :return: Hidden layer size.
        """
        return self.__hidden_layers[index]

    def getHiddenLayers(self) -> List[int]:
        """
        Getter for all hidden layers.

        :return: List of hidden layer sizes.
        """
        return self.__hidden_layers

    def getFunctions(self) -> List[Function]:
        """
        Getter for activation functions.

        :return: List of activation functions.
        """
        return self.__functions