from abc import ABC, abstractmethod
from Math.Matrix import Matrix
from Classification.Parameter.ActivationFunction import ActivationFunction
from Classification.Parameter.DeepNetworkParameter import DeepNetworkParameter

from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus
from SequenceProcessing.Initializer.Initializer import Initializer
import numpy as np
import pickle


class Model(ABC):
    corpus: SequenceCorpus
    layers: [Matrix]
    oldLayers: [Matrix]
    weights: [Matrix]
    recurrentWeights: [Matrix]
    classLabels: [str]
    activationFunction: ActivationFunction
    parameters: DeepNetworkParameter

    def __init__(self):
        self.corpus = None
        self.layers = []
        self.oldLayers = []
        self.weights = []
        self.recurrentWeights = []
        self.classLabels = []
        self.activationFunction = None
        self.parameters = None

    def train(self, corpus: SequenceCorpus, parameters: DeepNetworkParameter, initializer: Initializer):
        self.parameters = parameters
        self.corpus = corpus
        self.activationFunction = parameters.getActivationFunction()
        self.classLabels = corpus.getClassLabels()

        inputSize = corpus.getSentence(0).getWord(0).getVector().size()
        self.layers.append(Matrix(inputSize, 1))

        for i in range(parameters.layerSize()):
            self.oldLayers.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.layers.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.recurrentWeights.append(initializer.initialize(
                parameters.getHiddenNodes(i),
                parameters.getHiddenNodes(i),
                parameters.getSeed()
            ))

        self.layers.append(Matrix(len(self.classLabels), 1))

        for i in range(len(self.layers) - 1):
            self.weights.append(initializer.initialize(
                self.layers[i + 1].getRow(),
                self.layers[i].getRow() + 1,
                parameters.getSeed()
            ))

        epoch = parameters.getEpoch()
        learningRate = parameters.getLearningRate()

        for e in range(epoch):
            print(f"Epoch: {e + 1}")
            corpus.shuffleSentences(parameters.getSeed())

            for j in range(corpus.sentenceCount()):
                sentence = corpus.getSentence(j)
                for k in range(sentence.wordCount()):
                    self.calculateOutput(sentence, k)
                    self.backpropagation(sentence, k, learningRate)
                    self.clear()
                self.clearOldValues()

            learningRate *= parameters.getEtaDecrease()

    def createInputVector(self, word: LabelledVectorizedWord):
        for i in range(self.layers[0].getRow()):
            self.layers[0].setValue(i, 0, word.getVector().getValue(i))
        self.layers[0] = self.biased(self.layers[0])

    def biased(self, m: Matrix) -> Matrix:
        biasedMatrix = Matrix(m.getRow() + 1, m.getColumn())
        for i in range(m.getRow()):
            biasedMatrix.setValue(i, 0, m.getValue(i, 0))
        biasedMatrix.setValue(m.getRow(), 0, 1.0)
        return biasedMatrix

    def oldLayersUpdate(self):
        for i in range(len(self.oldLayers)):
            for j in range(self.oldLayers[i].getRow()):
                self.oldLayers[i].setValue(j, 0, self.layers[i + 1].getValue(j, 0))

    def setLayersValuesToZero(self):
        for j in range(len(self.layers) - 1):  # Loop through all layers except the last
            size = self.layers[j].getRow()
            self.layers[j] = Matrix(size - 1, 1)  # Reinitialize the layer as a new matrix
            for i in range(self.layers[j].getRow()):  # Set all values in the matrix to 0.0
                self.layers[j].setValue(i, 0, 0.0)

        # For the last layer, set all its values to 0.0
        for i in range(self.layers[len(self.layers) - 1].getRow()):
            self.layers[len(self.layers) - 1].setValue(i, 0, 0.0)

    def calculateOneMinusMatrix(self, hidden: Matrix) -> Matrix:
        oneMinus = Matrix(hidden.getRow(), 1)
        for i in range(hidden.getRow()):
            oneMinus.setValue(i, 0, 1 - hidden.getValue(i, 0))
        return oneMinus

    def normalizeOutput(self):
        expValues = [np.exp(self.layers[len(self.layers) - 1].getValue(i, 0)) for i in
                     range(self.layers[len(self.layers) - 1].getRow())]
        total = sum(expValues)
        for i in range(self.layers[len(self.layers) - 1].getRow()):
            self.layers[len(self.layers) - 1].setValue(i, 0, expValues[i] / total)

    def calculateRMinusY(self, word: LabelledVectorizedWord) -> Matrix:
        r = Matrix(len(self.classLabels), 1)  # or self.classLabels.size()?
        index = self.classLabels.index(word.getClassLabel())
        r.setValue(index, 0, 1.0)

        for i in range(len(self.classLabels)):
            r.setValue(i, 0, r.getValue(i, 0) - self.layers[len(self.layers) - 1].getValue(i, 0))

        return r

    def derivative(self, matrix: Matrix, function: ActivationFunction) -> Matrix:
        if function == ActivationFunction.SIGMOID:
            # Calculate SIGMOID derivative
            oneMinusHidden = self.calculateOneMinusMatrix(matrix)
            return matrix.elementProduct(oneMinusHidden)

        elif function == ActivationFunction.TANH:
            # Calculate TANH derivative
            oneMinusA2 = Matrix(matrix.getRow(), 1)  # Initialize a new matrix
            a2 = matrix.elementProduct(matrix)
            for i in range(oneMinusA2.getRow()):
                oneMinusA2.setValue(i, 0, 1.0 - a2.getValue(i, 0))
            return oneMinusA2

        elif function == ActivationFunction.RELU:
            # Calculate RELU derivative
            der = Matrix(matrix.getRow(), 1)  # Initialize a new matrix
            for i in range(matrix.getRow()):
                if matrix.getValue(i, 0) > 0:
                    der.setValue(i, 0, 1.0)  # Set derivative to 1.0 for positive values
                else:
                    der.setValue(i, 0, 0.0)  # Set derivative to 0.0 for non-positive values
            return der

        else:
            raise ValueError("Unsupported ActivationFunction type OR MatrixDimensionMismatch")

    def calculateActivationFunction(self, matrix: Matrix, function: ActivationFunction) -> Matrix:
        result = Matrix(matrix.getRow(), matrix.getColumn())  # Initialize a new matrix with the same dimensions

        if function == ActivationFunction.SIGMOID:
            for i in range(matrix.getRow()):
                result.setValue(i, 0, 1 / (1 + np.exp(-matrix.getValue(i, 0))))

        elif function == ActivationFunction.RELU:
            for i in range(matrix.getRow()):
                if matrix.getValue(i, 0) < 0:
                    result.setValue(i, 0, 0.0)
                else:
                    result.setValue(i, 0, matrix.getValue(i, 0))

        elif function == ActivationFunction.TANH:
            for i in range(matrix.getRow()):
                result.setValue(i, 0, np.tanh(matrix.getValue(i, 0)))

        else:
            raise ValueError("Unsupported ActivationFunction type")

        return result

    def clear(self):
        self.oldLayersUpdate()
        self.setLayersValuesToZero()

    def clearOldValues(self):
        for oldLayer in self.oldLayers:  # Loop through each oldLayer in oldLayers
            for k in range(oldLayer.getRow()):  # Iterate through all rows in the matrix
                oldLayer.setValue(k, 0, 0.0)  # Set the value at (k, 0) to 0.0

    @abstractmethod
    def calculateOutput(self, sentence, index):
        pass

    @abstractmethod
    def backpropagation(self, sentence, index, learningRate):
        pass

    def predict(self, sentence) -> [str]:
        predictions = []  # To store predicted class labels

        for i in range(sentence.wordCount()):  # Loop through each word in the sentence
            self.calculateOutput(sentence, i)  # Calculate the output for the current word

            # Find the index of the maximum value in the last layer
            maxIndex = max(
                range(self.layers[len(self.layers) - 1].getRow()),  # Iterate over the rows of the last layer
                key=lambda j: self.layers[len(self.layers) - 1].getValue(j, 0)  # Use the value at (j, 0) for comparison
            )

            # Append the corresponding class label
            predictions.append(self.classLabels[maxIndex])

            # Clear the state after processing each word
            self.clear()

        # Clear old values after processing the entire sentence
        self.clearOldValues()

        return predictions  # Return the list of predicted class labels

    def save(self, fileName: str):
        with open(fileName, 'wb') as file:
            pickle.dump(self, file)
