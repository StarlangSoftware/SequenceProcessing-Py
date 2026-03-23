from typing import List
import random

from ComputationalGraph.Function.Negation import Negation
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Function.Tanh import Tanh
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.ConcatenatedNode import ConcatenatedNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode

from SequenceProcessing.Functions.AdditionByConstant import AdditionByConstant
from SequenceProcessing.Functions.RemoveBias import RemoveBias
from SequenceProcessing.Functions.Switch import Switch

from Math.Tensor import Tensor
from SequenceProcessing.Parameters.RecurrentNeuralNetworkParameter import RecurrentNeuralNetworkParameter

from SequenceProcessing.Classification.RecurrentNeuralNetworkModel import RecurrentNeuralNetworkModel


class GatedRecurrentUnitModel(RecurrentNeuralNetworkModel):
    """
    Gated Recurrent Unit (GRU) model implementation.
    """

    __switches: List[Switch]

    def __init__(self, parameter: NeuralNetworkParameter, word_embedding_length: int):
        """
        Constructor for GRU model.

        :param parameter: Neural network parameters.
        :param word_embedding_length: Word embedding size.
        """
        super().__init__(parameter, word_embedding_length)
        self.__switches = []

    def train(self, train_set: List[Tensor]) -> None:
        """
        Trains the GRU model.

        :param train_set: Training dataset.
        """
        random_generator = random.Random(self.parameters.getSeed())
        time_step = self.findTimeStep(train_set)

        weights = []
        recurrent_weights = []

        current_length = self.wordEmbeddingLength + 1

        # Initialize weights
        for i in range(self.parameters.size()):
            for j in range(3):
                w = Tensor(
                    self.parameters.initializeWeights(
                        current_length,
                        self.parameters.getHiddenLayer(i),
                        random_generator
                    ),
                    (current_length, self.parameters.getHiddenLayer(i))
                )
                weights.append(MultiplicationNode(w))

                rw = Tensor(
                    self.parameters.initializeWeights(
                        self.parameters.getHiddenLayer(i),
                        self.parameters.getHiddenLayer(i),
                        random_generator
                    ),
                    (self.parameters.getHiddenLayer(i), self.parameters.getHiddenLayer(i))
                )
                recurrent_weights.append(MultiplicationNode(rw))

            current_length = self.parameters.getHiddenLayer(i) + 1

        # Output layer weights
        weights.append(MultiplicationNode(
            Tensor(
                self.parameters.initializeWeights(
                    current_length,
                    self.parameters.getClassLabelSize(),
                    random_generator
                ),
                (current_length, self.parameters.getClassLabelSize())
            )
        ))

        current_old_layers = []
        output_nodes = []

        for k in range(time_step):
            self.__switches.append(Switch())

            new_old_layers = []

            input_node = MultiplicationNode(False, True)
            self.inputNodes.append(input_node)

            current = input_node

            for i in range(self.parameters.size()):
                if current_old_layers:
                    aw = self.addEdge(current, weights[i * 3])

                    o_without_bias = self.addEdge(current_old_layers[i], RemoveBias())
                    ou = self.addEdge(o_without_bias, recurrent_weights[i * 3])

                    aw_ou = self.addAdditionEdge(aw, ou, False)

                    zt = self.addEdge(aw_ou, self.parameters.getActivationFunction(i * 2))

                    aw = self.addEdge(current, weights[i * 3 + 1])
                    ou = self.addEdge(o_without_bias, recurrent_weights[i * 3 + 1])

                    aw_ou = self.addAdditionEdge(aw, ou, False)
                    rt = self.addEdge(aw_ou, self.parameters.getActivationFunction(i * 2 + 1))

                    aw = self.addEdge(current, weights[i * 3 + 2])

                    rt_ht1 = self.addEdge(rt, o_without_bias, False, True)
                    ou = self.addEdge(rt_ht1, recurrent_weights[i * 3 + 2])

                    aw_ou = self.addAdditionEdge(aw, ou, False)

                    h_temp = self.addEdge(aw_ou, Tanh())

                    minus_zt = self.addEdge(zt, Negation())
                    one_minus_zt = self.addEdge(minus_zt, AdditionByConstant(1.0))

                    aw = self.addEdge(one_minus_zt, o_without_bias, False, True)
                    ou = self.addEdge(h_temp, zt, False, True)

                    a_function = self.addAdditionEdge(aw, ou, True)

                else:
                    aw = self.addEdge(current, weights[i * 3])

                    zt = self.addEdge(aw, self.parameters.getActivationFunction(i * 2))

                    aw = self.addEdge(current, weights[i * 3 + 2])
                    h_temp = self.addEdge(aw, Tanh())

                    a_function = self.addEdge(zt, h_temp, True, True)

                current = a_function
                new_old_layers.append(a_function)

            current_old_layers = new_old_layers

            node = self.addEdge(current, weights[-1])
            output_nodes.append(self.addEdge(node, self.__switches[k]))

        concatenated_node = self.concatEdges(output_nodes, 0)
        self.outputNode = self.addEdge(concatenated_node, Softmax())

        class_label_node = ComputationalNode()
        self.inputNodes.append(class_label_node)

        loss_inputs = [self.outputNode, class_label_node]

        self.addFunctionEdge(loss_inputs, self.parameters.getLossFunction(), False)

        super().train(train_set, random_generator)