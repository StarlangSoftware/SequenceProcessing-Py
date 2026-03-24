from typing import List
import random

from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.ConcatenatedNode import ConcatenatedNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode

from Math.Tensor import Tensor

from SequenceProcessing.Functions.RemoveBias import RemoveBias
from SequenceProcessing.Functions.Switch import Switch
from SequenceProcessing.Parameters.RecurrentNeuralNetworkParameter import RecurrentNeuralNetworkParameter


class RecurrentNeuralNetworkModel(ComputationalGraph):
    """
    Base class for recurrent neural network models.
    """

    wordEmbeddingLength: int
    __switches: List[Switch]

    def __init__(self, parameter: NeuralNetworkParameter, word_embedding_length: int):
        """
        Constructor.

        :param parameter: Neural network parameters.
        :param word_embedding_length: Embedding size.
        """
        super().__init__(parameter)
        self.wordEmbeddingLength = word_embedding_length
        self.__switches = []

    def getSwitches(self) -> List[Switch]:
        """
        Getter for switches.
        """
        return self.__switches

    def setSwitches(self, switches: List[Switch]) -> None:
        """
        Setter for switches.
        """
        self.__switches = switches

    def createInputTensors(self, instance: Tensor) -> List[int]:
        """
        Creates input tensors from sequence.

        :param instance: Input tensor.
        :return: Class labels.
        """
        class_labels = []
        time_step = instance.getShape()[0] // (self.wordEmbeddingLength + 1)

        j = 0

        for i in range(len(self.input_nodes) - 1):
            if i < time_step:
                self.__switches[i].setTurn(True)

                values = []
                for _ in range(self.wordEmbeddingLength):
                    values.append(instance.getValue((j,)))
                    j += 1

                class_labels.append(int(instance.getValue((j,))))
                j += 1

                self.input_nodes[i].setValue(
                    Tensor(values, (1, len(values)))
                )

            else:
                self.__switches[i].setTurn(False)

                values = [0.0] * self.wordEmbeddingLength

                class_labels.append(0)
                j += 1

                self.input_nodes[i].setValue(
                    Tensor(values, (1, len(values)))
                )

        return class_labels

    def trainInternal(self, train_set: List[Tensor], random_generator: random.Random) -> None:
        """
        Internal training loop.

        :param train_set: Training set.
        :param random_generator: Random generator.
        """
        for _ in range(self.parameters.getEpoch()):

            # Shuffle
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            for instance in train_set:
                class_labels = self.createInputTensors(instance)

                class_label_values = []

                for class_label in class_labels:
                    for i in range(self.parameters.getClassLabelSize()):
                        if i == class_label:
                            class_label_values.append(1.0)
                        else:
                            class_label_values.append(0.0)

                self.input_nodes[-1].setValue(
                    Tensor(
                        class_label_values,
                        (len(class_labels), self.parameters.getClassLabelSize())
                    )
                )

                self.forwardCalculation()
                self.backpropagation()

            self.parameters.getOptimizer().setLearningRate()

    def findTimeStep(self, train_set: List[Tensor]) -> int:
        """
        Finds max sequence length.

        :param train_set: Training set.
        :return: Time step.
        """
        time_step = -1

        for tensor in train_set:
            size = tensor.getShape()[0]
            current = size // (self.wordEmbeddingLength + 1)

            if time_step < current:
                time_step = current

        return time_step

    def train(self, train_set: List[Tensor]) -> None:
        """
        Many-to-many RNN training.
        """
        random_generator = random.Random(self.parameters.getSeed())

        time_step = self.findTimeStep(train_set)

        weights = []
        recurrent_weights = []

        current_length = self.wordEmbeddingLength + 1
        params = self.parameters

        for i in range(params.size()):
            weights.append(
                MultiplicationNode(
                    Tensor(
                        params.initializeWeights(
                            current_length,
                            params.getHiddenLayer(i),
                            random_generator
                        ),
                        (current_length, params.getHiddenLayer(i))
                    )
                )
            )

            recurrent_weights.append(
                MultiplicationNode(
                    Tensor(
                        params.initializeWeights(
                            params.getHiddenLayer(i),
                            params.getHiddenLayer(i),
                            random_generator
                        ),
                        (params.getHiddenLayer(i), params.getHiddenLayer(i))
                    )
                )
            )

            current_length = params.getHiddenLayer(i) + 1

        weights.append(
            MultiplicationNode(
                Tensor(
                    params.initializeWeights(
                        current_length,
                        params.getClassLabelSize(),
                        random_generator
                    ),
                    (current_length, params.getClassLabelSize())
                )
            )
        )

        current_old_layers = []
        output_nodes = []

        for k in range(time_step):
            self.__switches.append(Switch())

            new_old_layers = []

            input_node = MultiplicationNode(False, True)
            self.input_nodes.append(input_node)

            current = input_node

            for i in range(params.size()):
                if current_old_layers:
                    aw = self.addEdge(current, weights[i])

                    o_without_bias = self.addEdge(
                        current_old_layers[i],
                        RemoveBias()
                    )

                    ou = self.addEdge(o_without_bias, recurrent_weights[i])

                    a = self.addAdditionEdge(aw, ou, False)

                    a_function = self.addEdge(
                        a,
                        params.getActivationFunction(i),
                        True
                    )
                else:
                    aw = self.addEdge(current, weights[i], False)

                    a_function = self.addEdge(
                        aw,
                        params.getActivationFunction(i),
                        True
                    )

                current = a_function
                new_old_layers.append(a_function)

            current_old_layers = new_old_layers

            node = self.addEdge(current, weights[-1])

            output_nodes.append(
                self.addEdge(node, self.__switches[k])
            )

        concatenated_node = self.concatEdges(output_nodes, 0)

        self.output_node = self.addEdge(concatenated_node, Softmax())

        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)

        loss_inputs = [self.output_node, class_label_node]

        self.addFunctionEdge(loss_inputs, params.getLossFunction(), False)

        self.trainInternal(train_set, random_generator)

    def getOutputValue(self, output_node: ComputationalNode) -> List[float]:
        """
        Extracts predicted class indices.
        """
        class_labels = []

        shape = output_node.getValue().getShape()

        for i in range(shape[0]):
            index = -1
            max_val = float("-inf")

            for j in range(shape[1]):
                val = output_node.getValue().getValue((i, j))
                if val > max_val:
                    max_val = val
                    index = j

            class_labels.append(float(index))

        return class_labels

    def test(self, test_set: List[Tensor]):
        """
        Evaluates model.
        """
        count = 0
        total = 0

        for instance in test_set:
            gold = self.createInputTensors(instance)
            pred = self.predict()

            time_step = instance.getShape()[0] // (self.wordEmbeddingLength + 1)

            for j in range(time_step):
                if gold[j] == int(pred[j]):
                    count += 1
                total += 1

        return count / total