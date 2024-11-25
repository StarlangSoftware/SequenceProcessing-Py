from Math.Matrix import Matrix
from Classification.Parameter.ActivationFunction import ActivationFunction
from SequenceProcessing.Sequence.SequenceCorpus import SequenceCorpus
from SequenceProcessing.Initializer.Initializer import Initializer
from Model import Model


class LongShortTermMemoryModel(Model):
    def __init__(self):
        super().__init__()
        self.fVectors = []
        self.fWeights = []
        self.fRecurrentWeights = []
        self.gVectors = []
        self.gWeights = []
        self.gRecurrentWeights = []
        self.iVectors = []
        self.iWeights = []
        self.iRecurrentWeights = []
        self.oVectors = []
        self.oWeights = []
        self.oRecurrentWeights = []
        self.cVectors = []
        self.cOldVectors = []

    def train(self, corpus: SequenceCorpus, parameters, initializer: Initializer):
        layers = [corpus.getSentence(0).getWord(0).getVector().size()]
        for i in range(parameters.layerSize()):
            layers.append(parameters.getHiddenNodes(i))
        layers.append(len(corpus.getClassLabels()))

        for i in range(parameters.layerSize()):
            self.fVectors.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.gVectors.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.iVectors.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.oVectors.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.cVectors.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.cOldVectors.append(Matrix(parameters.getHiddenNodes(i), 1))

            self.fWeights.append(initializer.initialize(layers[i + 1], layers[i] + 1, parameters.getSeed()))
            self.gWeights.append(initializer.initialize(layers[i + 1], layers[i] + 1, parameters.getSeed()))
            self.iWeights.append(initializer.initialize(layers[i + 1], layers[i] + 1, parameters.getSeed()))
            self.oWeights.append(initializer.initialize(layers[i + 1], layers[i] + 1, parameters.getSeed()))

            self.fRecurrentWeights.append(
                initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), parameters.getSeed())
            )
            self.gRecurrentWeights.append(
                initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), parameters.getSeed())
            )
            self.iRecurrentWeights.append(
                initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), parameters.getSeed())
            )
            self.oRecurrentWeights.append(
                initializer.initialize(parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), parameters.getSeed())
            )

        super().train(corpus, parameters, initializer)

    def calculateOutput(self, sentence, index):
        word = sentence.getWord(index)
        self.createInputVector(word)

        kVectors = []
        jVectors = []

        for i in range(len(self.layers) - 2):
            # Forget Gate
            self.fVectors[i] = self.calculateActivationFunction(
                self.fRecurrentWeights[i].multiply(self.oldLayers[i])
                + self.fWeights[i].multiply(self.layers[i]),
                self.activationFunction,
            )

            # Memory Cell Update
            kVectors.append(self.cOldVectors[i].elementProduct(self.fVectors[i]))
            self.gVectors[i] = self.calculateActivationFunction(
                self.gRecurrentWeights[i].multiply(self.oldLayers[i])
                + self.gWeights[i].multiply(self.layers[i]),
                ActivationFunction.TANH,
            )
            self.iVectors[i] = self.calculateActivationFunction(
                self.iRecurrentWeights[i].multiply(self.oldLayers[i])
                + self.iWeights[i].multiply(self.layers[i]),
                self.activationFunction,
            )
            jVectors.append(self.gVectors[i].elementProduct(self.iVectors[i]))

            self.cVectors[i] = kVectors[i] + jVectors[i]

            # Output Gate
            self.oVectors[i] = self.calculateActivationFunction(
                self.oRecurrentWeights[i].multiply(self.oldLayers[i])
                + self.oWeights[i].multiply(self.layers[i]),
                self.activationFunction,
            )

            self.layers[i + 1] = self.oVectors[i].elementProduct(
                self.calculateActivationFunction(self.cVectors[i], ActivationFunction.TANH)
            )
            self.layers[i + 1] = self.biased(self.layers[i + 1])

        self.layers[-1] += self.weights[-1].multiply(self.layers[-2])
        self.normalizeOutput()

    def oldLayersUpdate(self):
        for i in range(len(self.oldLayers)):
            for j in range(self.oldLayers[i].getRow()):
                self.oldLayers[i].setValue(j, 0, self.layers[i + 1].getValue(j, 0))
                self.cOldVectors[i].setValue(j, 0, self.cVectors[i].getValue(j, 0))

    def backpropagation(self, sentence, index, learningRate):
        word = sentence.getWord(index)
        rMinusY = self.calculateRMinusY(word)
        rMinusY.multiplyWithConstant(learningRate)

        deltaWeight = rMinusY.multiply(self.layers[-2].transpose())

        # Update weights and recurrent weights
        for i in range(len(self.fWeights)):
            self.fWeights[i] += deltaWeight
            self.gWeights[i] += deltaWeight
            self.iWeights[i] += deltaWeight
            self.oWeights[i] += deltaWeight

            self.fRecurrentWeights[i] += deltaWeight
            self.gRecurrentWeights[i] += deltaWeight
            self.iRecurrentWeights[i] += deltaWeight
            self.oRecurrentWeights[i] += deltaWeight

    def clear(self):
        super().clear()
        for l in range(len(self.layers) - 2):
            for m in range(self.fVectors[l].getRow()):
                self.fVectors[l].setValue(m, 0, 0.0)
                self.gVectors[l].setValue(m, 0, 0.0)
                self.iVectors[l].setValue(m, 0, 0.0)
                self.oVectors[l].setValue(m, 0, 0.0)
                self.cVectors[l].setValue(m, 0, 0.0)

    def clearOldValues(self):
        for i in range(len(self.oldLayers)):
            for k in range(self.oldLayers[i].getRow()):
                self.oldLayers[i].setValue(k, 0, 0.0)
                self.cOldVectors[i].setValue(k, 0, 0.0)
