from Math.Matrix import Matrix

from SequenceProcessing.Classification.Model import Model
from Classification.Parameter.ActivationFunction import ActivationFunction


class GatedRecurrentUnitModel(Model):

    def __init__(self):
        super().__init__()
        self.aVectors = []
        self.zVectors = []
        self.rVectors = []
        self.zWeights = []
        self.rRecurrentWeights = []
        self.rWeights = []
        self.zRecurrentWeights = []

    def train(self, corpus, parameters, initializer):

        seed = parameters.getSeed()

        # Initialize the layers list
        layers = [corpus.getSentence(0).getWord(0).getVector().size()]

        for i in range(parameters.layerSize()):
            layers.append(parameters.getHiddenNodes(i))

        layers.append(len(corpus.getClassLabels()))

        # Initialize lists for vectors and weights
        self.aVectors = []
        self.zVectors = []
        self.rVectors = []
        self.zWeights = []
        self.zRecurrentWeights = []
        self.rWeights = []
        self.rRecurrentWeights = []

        for i in range(parameters.layerSize()):
            # Create matrices for vectors
            self.aVectors.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.zVectors.append(Matrix(parameters.getHiddenNodes(i), 1))
            self.rVectors.append(Matrix(parameters.getHiddenNodes(i), 1))

            # Initialize weights using the provided initializer
            self.zWeights.append(initializer.initialize(
                layers[i + 1], layers[i] + 1, seed)
            )
            self.rWeights.append(initializer.initialize(
                layers[i + 1], layers[i] + 1, seed)
            )
            self.zRecurrentWeights.append(initializer.initialize(
                parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), seed)
            )
            self.rRecurrentWeights.append(initializer.initialize(
                parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), seed)
            )

        # Call the superclass train method
        super().train(corpus, parameters, initializer)

    def clear(self):
        super().clear()
        for i in range(len(self.layers) - 2):
            for m in range(self.aVectors[i].getRow()):
                self.aVectors[i].setValue(m, 0, 0.0)
                self.zVectors[i].setValue(m, 0, 0.0)
                self.rVectors[i].setValue(m, 0, 0.0)

    def calculateOutput(self, sentence, index):
        word = sentence.getWord(index)
        self.createInputVector(word)
        for i in range(len(self.layers) - 2):
            self.rVectors[i].append(self.rWeights[i].multiply(self.layers[i]))
            self.zVectors[i].append(self.zWeights[i].multiply(self.layers[i]))
            self.rVectors[i].append(self.rRecurrentWeights[i].multiply(self.oldLayers[i]))
            self.zVectors[i].append(self.zRecurrentWeights[i].multiply(self.oldLayers[i]))

            self.rVectors[i] = self.calculateActivationFunction(self.rVectors[i], self.activationFunction)
            self.zVectors[i] = self.calculateActivationFunction(self.zVectors[i], self.activationFunction)

            self.aVectors[i].append(
                self.recurrentWeights[i].multiply(
                    self.rVectors[i].elementProduct(self.oldLayers[i])
                )
            )
            self.aVectors[i].append(self.weights[i].multiply(self.layers[i]))

            self.aVectors[i] = self.calculateActivationFunction(self.aVectors[i], ActivationFunction.TANH)

            self.layers[i + 1].append(
                self.calculateOneMinusMatrix(self.zVectors[i]).elementProduct(self.oldLayers[i])
            )
            self.layers[i + 1].append(self.zVectors[i].elementProduct(self.aVectors[i]))
            self.layers[i + 1] = self.biased(self.layers[i + 1])

        self.layers[-1].append(
            self.weights[-1].multiply(self.layers[-2])
        )
        self.normalizeOutput()

    def backpropagation(self, sentence, index, learningRate):
        word = sentence.getWord(index)
        rMinusY = self.calculateRMinusY(word)
        rMinusY.multiplyWithConstant(learningRate)

        deltaWeights = []
        deltaRecurrentWeights = []
        rDeltaWeights = []
        rDeltaRecurrentWeights = []
        zDeltaWeights = []
        zDeltaRecurrentWeights = []

        deltaWeights.append(rMinusY.multiply(self.layers[-2].transpose()))
        deltaWeights.append(
            rMinusY.transpose()
            .multiply(
                self.weights[-1].partial(0, self.weights[-1].getRow() - 1, 0, self.weights[-1].getColumn() - 2)
            )
            .transpose()
        )
        deltaRecurrentWeights.append(deltaWeights[-1].clone())
        rDeltaWeights.append(deltaWeights[-1].clone())
        rDeltaRecurrentWeights.append(deltaWeights[-1].clone())
        zDeltaWeights.append(deltaWeights[-1].clone())
        zDeltaRecurrentWeights.append(deltaWeights[-1].clone())

        for i in range(self.parameters.layerSize() - 1, -1, -1):
            delta = deltaWeights[-1].elementProduct(self.zVectors[i]).elementProduct(
                self.derivative(self.aVectors[i], ActivationFunction.TANH)
            )
            zDelta = zDeltaWeights[-1].elementProduct(
                self.aVectors[i].difference(self.oldLayers[i])
            ).elementProduct(
                self.derivative(self.zVectors[i], self.activationFunction)
            )
            rDelta = (
                rDeltaWeights[-1]
                .elementProduct(self.aVectors[i].difference(self.oldLayers[i]))
                .elementProduct(self.derivative(self.zVectors[i], self.activationFunction))
                .transpose()
                .multiply(self.recurrentWeights[i])
                .transpose()
                .elementProduct(self.oldLayers[i])
                .elementProduct(self.derivative(self.rVectors[i], self.activationFunction))
            )

            deltaWeights[-1] = delta.multiply(self.layers[i].transpose())
            deltaRecurrentWeights[-1] = delta.multiply(
                self.rVectors[i].elementProduct(self.oldLayers[i]).transpose()
            )
            zDeltaWeights[-1] = zDelta.multiply(self.layers[i].transpose())
            zDeltaRecurrentWeights[-1] = zDelta.multiply(self.oldLayers[i].transpose())
            rDeltaWeights[-1] = rDelta.multiply(self.layers[i].transpose())
            rDeltaRecurrentWeights[-1] = rDelta.multiply(self.oldLayers[i].transpose())

            if i > 0:
                deltaWeights.append(
                    delta.transpose()
                    .multiply(
                        self.weights[i].partial(
                            0, self.weights[i].getRow() - 1, 0, self.weights[i].getColumn() - 2
                        )
                    )
                    .transpose()
                )
                deltaRecurrentWeights.append(deltaWeights[-1].clone())
                zDeltaWeights.append(
                    zDelta.transpose()
                    .multiply(
                        self.zWeights[i].partial(
                            0, self.zWeights[i].getRow() - 1, 0, self.zWeights[i].getColumn() - 2
                        )
                    )
                    .transpose()
                )
                zDeltaRecurrentWeights.append(zDeltaWeights[-1].clone())
                rDeltaWeights.append(
                    rDelta.transpose()
                    .multiply(
                        self.rWeights[i].partial(
                            0, self.rWeights[i].getRow() - 1, 0, self.rWeights[i].getColumn() - 2
                        )
                    )
                    .transpose()
                )
                rDeltaRecurrentWeights.append(rDeltaWeights[-1].clone())

        self.weights[-1].append(deltaWeights[0])
        deltaWeights.pop(0)

        for i in range(len(deltaWeights)):
            self.weights[-i - 2].append(deltaWeights[i])
            self.rWeights[-i - 1].append(rDeltaWeights[i])
            self.zWeights[-i - 1].append(zDeltaWeights[i])
            self.recurrentWeights[-i - 1].append(deltaRecurrentWeights[i])
            self.zRecurrentWeights[-i - 1].append(zDeltaRecurrentWeights[i])
            self.rRecurrentWeights[-i - 1].append(rDeltaRecurrentWeights[i])
