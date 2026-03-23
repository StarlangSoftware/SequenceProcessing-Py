from typing import List
import random

from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Function.Tanh import Tanh
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.ConcatenatedNode import ConcatenatedNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode

from Math.Tensor import Tensor

from SequenceProcessing.Functions.RemoveBias import RemoveBias
from SequenceProcessing.Functions.Switch import Switch
from SequenceProcessing.Parameters.RecurrentNeuralNetworkParameter import RecurrentNeuralNetworkParameter
from SequenceProcessing.Classification.RecurrentNeuralNetworkModel import RecurrentNeuralNetworkModel


class LongShortTermMemoryModel(RecurrentNeuralNetworkModel):
    """
    Long Short-Term Memory model implementation.
    """

    __switches: List[Switch]

    def __init__(self, parameter: NeuralNetworkParameter, word_embedding_length: int):
        """
        Constructor for LongShortTermMemoryModel.

        :param parameter: Neural network parameter object.
        :param word_embedding_length: Length of word embeddings.
        """
        super().__init__(parameter, word_embedding_length)
        self.__switches = []

    def getSwitches(self) -> List[Switch]:
        """
        Getter for switches.

        :return: List of switch nodes.
        """
        return self.__switches

    def setSwitches(self, switches: List[Switch]) -> None:
        """
        Setter for switches.

        :param switches: New list of switch nodes.
        """
        self.__switches = switches

    def train(self, train_set: List[Tensor]) -> None:
        """
        Trains the LSTM model.

        :param train_set: Training set.
        """
        random_generator = random.Random(self.parameters.getSeed())
        time_step = self.findTimeStep(train_set)

        weights = []
        recurrent_weights = []

        current_length = self.wordEmbeddingLength + 1
        recurrent_parameter = self.parameters

        for i in range(recurrent_parameter.size()):
            for j in range(4):
                weights.append(
                    MultiplicationNode(
                        Tensor(
                            recurrent_parameter.initializeWeights(
                                current_length,
                                recurrent_parameter.getHiddenLayer(i),
                                random_generator
                            ),
                            (current_length, recurrent_parameter.getHiddenLayer(i))
                        )
                    )
                )

                recurrent_weights.append(
                    MultiplicationNode(
                        Tensor(
                            recurrent_parameter.initializeWeights(
                                recurrent_parameter.getHiddenLayer(i),
                                recurrent_parameter.getHiddenLayer(i),
                                random_generator
                            ),
                            (recurrent_parameter.getHiddenLayer(i),
                             recurrent_parameter.getHiddenLayer(i))
                        )
                    )
                )

            current_length = recurrent_parameter.getHiddenLayer(i) + 1

        weights.append(
            MultiplicationNode(
                Tensor(
                    recurrent_parameter.initializeWeights(
                        current_length,
                        recurrent_parameter.getClassLabelSize(),
                        random_generator
                    ),
                    (current_length, recurrent_parameter.getClassLabelSize())
                )
            )
        )

        current_old_layers = []
        current_old_c_values = []
        output_nodes = []

        for k in range(time_step):
            self.__switches.append(Switch())

            new_old_layers = []
            new_old_c_values = []

            input_node = MultiplicationNode(False, True)
            self.inputNodes.append(input_node)

            current = input_node

            for i in range(0, len(weights) - 1, 4):
                if len(current_old_layers) > 0:
                    aw = self.addEdge(current, weights[i])

                    o_without_bias = self.addEdge(current_old_layers[i // 4], RemoveBias())

                    ou = self.addEdge(o_without_bias, recurrent_weights[i])
                    aw_ou = self.addAdditionEdge(aw, ou, False)
                    it = self.addEdge(aw_ou, recurrent_parameter.getActivationFunction(i))

                    aw = self.addEdge(current, weights[i + 1])
                    ou = self.addEdge(o_without_bias, recurrent_weights[i + 1])
                    aw_ou = self.addAdditionEdge(aw, ou, False)
                    ft = self.addEdge(aw_ou, recurrent_parameter.getActivationFunction(i + 1))

                    aw = self.addEdge(current, weights[i + 2])
                    ou = self.addEdge(o_without_bias, recurrent_weights[i + 2])
                    aw_ou = self.addAdditionEdge(aw, ou, False)
                    ot = self.addEdge(aw_ou, recurrent_parameter.getActivationFunction(i + 2))

                    aw = self.addEdge(current, weights[i + 3])
                    ou = self.addEdge(o_without_bias, recurrent_weights[i + 3])
                    aw_ou = self.addAdditionEdge(aw, ou, False)
                    c_temp = self.addEdge(aw_ou, Tanh())

                    ft_ct1 = self.addEdge(ft, current_old_c_values[i // 4], False, True)
                    it_c_temp = self.addEdge(it, c_temp, False, True)
                    cmb = self.addAdditionEdge(ft_ct1, it_c_temp, False)

                    ct = self.addEdge(cmb, recurrent_parameter.getActivationFunction(i + 3))
                    tanh_ct = self.addEdge(ct, Tanh())
                    a_function = self.addEdge(tanh_ct, ot, True, True)
                else:
                    aw = self.addEdge(current, weights[i])
                    it = self.addEdge(aw, recurrent_parameter.getActivationFunction(i))

                    aw = self.addEdge(current, weights[i + 1])
                    ot = self.addEdge(aw, recurrent_parameter.getActivationFunction(i + 2))

                    aw = self.addEdge(current, weights[i + 3])
                    c_temp = self.addEdge(aw, Tanh())

                    it_c_temp = self.addEdge(it, c_temp, False, True)
                    ct = self.addEdge(it_c_temp, recurrent_parameter.getActivationFunction(i + 3))

                    tanh_ct = self.addEdge(ct, Tanh())
                    a_function = self.addEdge(tanh_ct, ot, True, True)

                current = a_function
                new_old_layers.append(a_function)
                new_old_c_values.append(ct)

            current_old_layers = new_old_layers
            current_old_c_values = new_old_c_values

            node = self.addEdge(current, weights[len(weights) - 1])
            output_nodes.append(self.addEdge(node, self.__switches[k]))

        concatenated_node = self.concatEdges(output_nodes, 0)
        self.outputNode = self.addEdge(concatenated_node, Softmax())

        class_label_node = ComputationalNode()
        self.inputNodes.append(class_label_node)

        loss_inputs = [self.outputNode, class_label_node]
        self.addFunctionEdge(loss_inputs, self.parameters.getLossFunction(), False)

        super().train(train_set, random_generator)