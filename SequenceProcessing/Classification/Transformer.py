from typing import List
import math
import random

from Classification.Performance.ClassificationPerformance import ClassificationPerformance
from ComputationalGraph.ComputationalGraph import ComputationalGraph
from ComputationalGraph.NeuralNetworkParameter import NeuralNetworkParameter
from ComputationalGraph.Function.Negation import Negation
from ComputationalGraph.Function.Softmax import Softmax
from ComputationalGraph.Node.ComputationalNode import ComputationalNode
from ComputationalGraph.Node.ConcatenatedNode import ConcatenatedNode
from ComputationalGraph.Node.MultiplicationNode import MultiplicationNode
from Dictionary.VectorizedDictionary import VectorizedDictionary
from Dictionary.VectorizedWord import VectorizedWord
from Math.Tensor import Tensor
from Math.Vector import Vector
from SequenceProcessing.Functions.Mean import Mean
from SequenceProcessing.Functions.Variance import Variance
from SequenceProcessing.Functions.SquareRoot import SquareRoot
from SequenceProcessing.Functions.Inverse import Inverse
from SequenceProcessing.Functions.Transpose import Transpose
from SequenceProcessing.Functions.MultiplyByConstant import MultiplyByConstant
from SequenceProcessing.Functions.Mask import Mask
from SequenceProcessing.Parameters.TransformerParameter import TransformerParameter


class Transformer(ComputationalGraph):
    """
    Transformer model implementation.
    """

    __dictionary: VectorizedDictionary
    __start_index: int
    __end_index: int

    def __init__(self,
                 parameter: NeuralNetworkParameter,
                 dictionary: VectorizedDictionary):
        """
        Constructor for Transformer.

        :param parameter: Neural network parameter object.
        :param dictionary: Vectorized dictionary.
        """
        super().__init__(parameter)
        self.__dictionary = dictionary
        self.__start_index = -1
        self.__end_index = -1

        for k in range(self.__dictionary.size()):
            if self.__dictionary.getWord(k).getName() == "<S>":
                self.__start_index = k
            elif self.__dictionary.getWord(k).getName() == "</S>":
                self.__end_index = k

    def getDictionary(self) -> VectorizedDictionary:
        """
        Getter for dictionary.

        :return: Dictionary.
        """
        return self.__dictionary

    def getStartIndex(self) -> int:
        """
        Getter for start index.

        :return: Start token index.
        """
        return self.__start_index

    def getEndIndex(self) -> int:
        """
        Getter for end index.

        :return: End token index.
        """
        return self.__end_index

    def positionalEncoding(self, tensor: Tensor, word_embedding_length: int) -> Tensor:
        """
        Applies positional encoding to the given tensor.

        :param tensor: Input tensor.
        :param word_embedding_length: Embedding length.
        :return: Positional encoded tensor.
        """
        values = []

        for i in range(tensor.getShape()[0]):
            for j in range(tensor.getShape()[1]):
                val = tensor.getValue((i, j))
                if j % 2 == 0:
                    values.append(
                        val + math.sin((i + 1.0) / math.pow(10000, j / word_embedding_length))
                    )
                else:
                    values.append(
                        val + math.cos((i + 1.0) / math.pow(10000, (j - 1.0) / word_embedding_length))
                    )

        return Tensor(values, tensor.getShape())

    def createInputTensors(self,
                           instance: Tensor,
                           input1: ComputationalNode,
                           input2: ComputationalNode,
                           word_embedding_length: int) -> List[int]:
        """
        Creates encoder and decoder input tensors.

        :param instance: Input tensor.
        :param input1: First input node.
        :param input2: Second input node.
        :param word_embedding_length: Embedding length.
        :return: Class label list.
        """
        is_output = False
        cur_length = 0
        class_labels = []
        values = []

        for i in range(instance.getShape()[0]):
            val = instance.getValue((i,))
            if val == float("inf"):
                is_output = True
                input1.setValue(Tensor(values, (cur_length // word_embedding_length, word_embedding_length)))
                input1.setValue(self.positionalEncoding(input1.getValue(), word_embedding_length))
                cur_length = 0
                values.clear()
            elif is_output:
                if (cur_length + 1) % (word_embedding_length + 1) == 0:
                    class_labels.append(int(val))
                else:
                    values.append(val)
                cur_length += 1
            else:
                values.append(val)
                cur_length += 1

        input2.setValue(Tensor(values, (len(values) // word_embedding_length, word_embedding_length)))
        input2.setValue(self.positionalEncoding(input2.getValue(), word_embedding_length))

        return class_labels

    def layerNormalization(self,
                           input_node: ComputationalNode,
                           parameter: TransformerParameter,
                           is_input: bool,
                           ln_size: List[int]) -> ComputationalNode:
        """
        Applies layer normalization.

        :param input_node: Input computational node.
        :param parameter: Transformer parameter object.
        :param is_input: Whether current normalization is input-side.
        :param ln_size: Layer normalization counters.
        :return: Normalized node.
        """
        data = []

        input_c1_mean = self.addEdge(input_node, Mean())
        mean1_minus = self.addEdge(input_c1_mean, Negation())
        input_c1_mean1_minus = self.addAdditionEdge(input_node, mean1_minus, False)

        variance1 = self.addEdge(input_c1_mean1_minus, Variance())
        root_variance1 = self.addEdge(variance1, SquareRoot(parameter.getEpsilon()))
        inverse_root_variance1 = self.addEdge(root_variance1, Inverse())

        ln_value1 = self.addEdge(input_c1_mean1_minus, inverse_root_variance1, False, True)

        if is_input:
            for _ in range(parameter.getL()):
                data.append(parameter.getGammaInputValue(ln_size[0]))
            ln_size[0] += 1
        else:
            for _ in range(parameter.getL()):
                data.append(parameter.getGammaOutputValue(ln_size[1]))
            ln_size[1] += 1

        gamma_input1 = MultiplicationNode(True, False, Tensor(data, (1, parameter.getL())), True)
        ln_value1_gamma_input1 = self.addEdge(ln_value1, gamma_input1)

        data.clear()

        if is_input:
            for _ in range(parameter.getL()):
                data.append(parameter.getBetaInputValue(ln_size[2]))
            ln_size[2] += 1
        else:
            for _ in range(parameter.getL()):
                data.append(parameter.getBetaOutputValue(ln_size[3]))
            ln_size[3] += 1

        beta_input1 = ComputationalNode(True, False, Tensor(data, (1, parameter.getL())))

        return self.addAdditionEdge(ln_value1_gamma_input1, beta_input1, False)

    def multiHeadAttention(self,
                           input_node: ComputationalNode,
                           parameter: TransformerParameter,
                           is_masked: bool,
                           random_generator: random.Random) -> List[ComputationalNode]:
        """
        Builds multi-head attention outputs.

        :param input_node: Input node.
        :param parameter: Transformer parameters.
        :param is_masked: Whether attention is masked.
        :param random_generator: Random generator.
        :return: Attention output nodes.
        """
        nodes = []

        for _ in range(parameter.getN()):
            wk = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            k = self.addEdge(input_node, wk)

            wq = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            q = self.addEdge(input_node, wq)

            wv = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            v = self.addEdge(input_node, wv)

            k_transpose = self.addEdge(k, Transpose())
            qk = self.addEdge(q, k_transpose, False, False)
            qk_dk = self.addEdge(qk, MultiplyByConstant(1.0 / math.sqrt(parameter.getDk())))

            if is_masked:
                m_qk_dk = self.addEdge(qk_dk, Mask())
                s_qk_dk = self.addEdge(m_qk_dk, Softmax())
            else:
                s_qk_dk = self.addEdge(qk_dk, Softmax())

            attention = self.addEdge(s_qk_dk, v)
            nodes.append(attention)

        return nodes

    def feedforwardNeuralNetwork(self,
                                 current: ComputationalNode,
                                 current_layer_size: int,
                                 parameter: TransformerParameter,
                                 random_generator: random.Random,
                                 is_input: bool) -> ComputationalNode:
        """
        Builds feedforward network block.

        :param current: Current node.
        :param current_layer_size: Current layer size.
        :param parameter: Transformer parameters.
        :param random_generator: Random generator.
        :param is_input: Whether current block is input-side.
        :return: Output node.
        """
        if is_input:
            size = parameter.getInputSize()
        else:
            size = parameter.getOutputSize()

        for i in range(size):
            if is_input:
                hidden_weight = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(
                            current_layer_size,
                            parameter.getInputHiddenLayer(i),
                            random_generator
                        ),
                        (current_layer_size, parameter.getInputHiddenLayer(i))
                    )
                )
                hidden_layer = self.addEdge(current, hidden_weight)
                current = self.addEdge(hidden_layer, parameter.getInputActivationFunction(i), True)
                current_layer_size = parameter.getInputHiddenLayer(i) + 1
            else:
                hidden_weight = MultiplicationNode(
                    Tensor(
                        parameter.initializeWeights(
                            current_layer_size,
                            parameter.getOutputHiddenLayer(i),
                            random_generator
                        ),
                        (current_layer_size, parameter.getOutputHiddenLayer(i))
                    )
                )
                hidden_layer = self.addEdge(current, hidden_weight)
                current = self.addEdge(hidden_layer, parameter.getOutputActivationFunction(i), True)
                current_layer_size = parameter.getOutputHiddenLayer(i) + 1

        output_weight = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(current_layer_size, parameter.getL(), random_generator),
                (current_layer_size, parameter.getL())
            )
        )
        output_layer = self.addEdge(current, output_weight)

        return self.addEdge(output_layer, Softmax())

    def train(self, train_set: List[Tensor]) -> None:
        """
        Trains the transformer model.

        :param train_set: Training set.
        """
        parameter = self.parameters
        ln_size = [0, 0, 0, 0]
        random_generator = random.Random(parameter.getSeed())

        # Encoder Block
        input1 = MultiplicationNode(False, True)
        self.input_nodes.append(input1)

        concatenated_node1 = self.concatEdges(
            self.multiHeadAttention(input1, parameter, False, random_generator),
            1
        )
        we = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(parameter.getL(), parameter.getL(), random_generator),
                (parameter.getL(), parameter.getL())
            )
        )
        c1 = self.addEdge(concatenated_node1, we)
        input_c1 = self.addAdditionEdge(input1, c1, False)
        y1 = self.layerNormalization(input_c1, parameter, True, ln_size)
        oe = self.addAdditionEdge(
            self.feedforwardNeuralNetwork(y1, parameter.getL(), parameter, random_generator, True),
            y1,
            False
        )
        encoder = self.layerNormalization(oe, parameter, True, ln_size)

        # Decoder Block
        input2 = MultiplicationNode(False, True)
        self.input_nodes.append(input2)

        concatenated_node2 = self.concatEdges(
            self.multiHeadAttention(input2, parameter, True, random_generator),
            1
        )
        wd1 = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(parameter.getL(), parameter.getL(), random_generator),
                (parameter.getL(), parameter.getL())
            )
        )
        c2 = self.addEdge(concatenated_node2, wd1)
        input_c2 = self.addAdditionEdge(input2, c2, False)
        cd2 = self.layerNormalization(input_c2, parameter, False, ln_size)

        nodes = []

        for _ in range(parameter.getN()):
            wk = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            k = self.addEdge(encoder, wk)

            wq = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            q = self.addEdge(cd2, wq)

            wv = MultiplicationNode(
                Tensor(
                    parameter.initializeWeights(parameter.getL(), parameter.getDk(), random_generator),
                    (parameter.getL(), parameter.getDk())
                )
            )
            v = self.addEdge(encoder, wv)

            k_transpose = self.addEdge(k, Transpose())
            qk = self.addEdge(q, k_transpose, False, False)
            qk_dk = self.addEdge(qk, MultiplyByConstant(1.0 / math.sqrt(parameter.getDk())))
            s_qk_dk = self.addEdge(qk_dk, Softmax())
            attention = self.addEdge(s_qk_dk, v)
            nodes.append(attention)

        concatenated_node3 = self.concatEdges(nodes, 1)
        wd2 = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(parameter.getL(), parameter.getL(), random_generator),
                (parameter.getL(), parameter.getL())
            )
        )
        cd3 = self.addEdge(concatenated_node3, wd2)
        cd3_cd2 = self.addAdditionEdge(cd2, cd3, False)
        yd1 = self.layerNormalization(cd3_cd2, parameter, False, ln_size)
        od = self.feedforwardNeuralNetwork(yd1, parameter.getL(), parameter, random_generator, False)
        oy = self.addAdditionEdge(od, yd1, False)
        d = self.layerNormalization(oy, parameter, False, ln_size)

        wdo = MultiplicationNode(
            Tensor(
                parameter.initializeWeights(parameter.getL(), parameter.getV(), random_generator),
                (parameter.getL(), parameter.getV())
            )
        )
        decoder = self.addEdge(d, wdo)
        self.output_node = self.addEdge(decoder, Softmax())

        class_label_node = ComputationalNode()
        self.input_nodes.append(class_label_node)

        loss_inputs = [self.output_node, class_label_node]
        self.addFunctionEdge(loss_inputs, parameter.getLossFunction(), False)

        # Training
        for _ in range(parameter.getEpoch()):
            for _ in range(len(train_set)):
                i1 = random_generator.randint(0, len(train_set) - 1)
                i2 = random_generator.randint(0, len(train_set) - 1)
                train_set[i1], train_set[i2] = train_set[i2], train_set[i1]

            for instance in train_set:
                class_labels = self.createInputTensors(
                    instance,
                    self.input_nodes[0],
                    self.input_nodes[1],
                    parameter.getL() - 1
                )

                class_label_values = []

                for class_label in class_labels:
                    for j in range(parameter.getV()):
                        if j == class_label:
                            class_label_values.append(1.0)
                        else:
                            class_label_values.append(0.0)

                self.input_nodes[2].setValue(
                    Tensor(class_label_values, (len(class_labels), parameter.getV()))
                )

                self.forwardCalculation()
                self.backpropagation()

            parameter.getOptimizer().setLearningRate()

    def setInputNode(self, bound: int, vector: Vector, node: ComputationalNode) -> None:
        """
        Sets the input node value with positional encoding.

        :param bound: Current bound value.
        :param vector: Input vector.
        :param node: Computational node.
        """
        data = []

        if node.getValue() is not None:
            data = list(node.getValue().getData())

        for i in range(vector.size()):
            if i % 2 == 0:
                data.append(
                    vector.getValue(i) + math.sin(bound / math.pow(10000, i / vector.size()))
                )
            else:
                data.append(
                    vector.getValue(i) + math.cos(bound / math.pow(10000, (i - 1.0) / vector.size()))
                )

        node.setValue(Tensor(data, (bound, vector.size())))

    def test(self, test_set: List[Tensor]) -> ClassificationPerformance:
        """
        Tests the transformer model.

        :param test_set: Test set.
        :return: Classification performance.
        """
        count = 0
        total = 0

        for instance in test_set:
            gold_class_labels = self.createInputTensors(
                instance,
                self.input_nodes[0],
                ComputationalNode(False, False),
                self.__dictionary.getWord(0).getVector().size()
            )

            j = 1
            current_word_index = self.__start_index

            while True:
                self.setInputNode(
                    j,
                    self.__dictionary.getWord(current_word_index).getVector(),
                    self.input_nodes[1]
                )

                class_labels = self.predict()

                if len(gold_class_labels) >= len(class_labels) and \
                        int(class_labels[-1]) == gold_class_labels[len(class_labels) - 1]:
                    count += 1

                total += 1
                j += 1
                current_word_index = int(class_labels[-1])

                if current_word_index == self.__end_index:
                    break

            if len(class_labels) < len(gold_class_labels):
                total += len(gold_class_labels) - len(class_labels)

        return ClassificationPerformance((count + 0.0) / total)

    def getOutputValue(self, computational_node: ComputationalNode) -> List[float]:
        """
        Extracts predicted class labels from output node.

        :param computational_node: Output node.
        :return: Predicted class labels.
        """
        class_labels = []
        value = computational_node.getValue()

        for i in range(value.getShape()[0]):
            max_val = float("-inf")
            index = -1.0

            for j in range(value.getShape()[1]):
                current = value.getValue((i, j))
                if current > max_val:
                    max_val = current
                    index = float(j)

            class_labels.append(index)

        return class_labels