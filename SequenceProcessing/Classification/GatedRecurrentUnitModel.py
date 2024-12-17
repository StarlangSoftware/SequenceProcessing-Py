from SequenceProcessing.Classification.Model import Model
from SequenceProcessing.Initializer.Initializer import Initializer
from SequenceProcessing.Sequence.LabelledVectorizedWord import LabelledVectorizedWord
from Corpus.Sentence import Sentence
from Classification.Parameter.ActivationFunction import ActivationFunction
from Classification.Parameter.DeepNetworkParameter import DeepNetworkParameter
import random
from Math import Matrix

class GatedRecurrentUnitModel(Model):
    
    def __init__(self):
        self.aVectors = []  
        self.zVectors = []
        self.rVectors = []
        self.zWeights = []
        self.rRecurrentWeights = []
        self.rWeights = []
        self.rRecurrentWeights = []
        
    def train(self, corpus, parameters, initializer):
        
        seed = parameters.getSeed()
        random.seed(seed)
        
        # Initialize the layers list
        layers = []
        layers.append(corpus.getSentence(0).getWord(0).getVector().size())
        
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
                layers[i + 1], layers[i] + 1, random)
            )
            self.rWeights.append(initializer.initialize(
                layers[i + 1], layers[i] + 1, random)
            )
            self.zRecurrentWeights.append(initializer.initialize(
                parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), random)
            )
            self.rRecurrentWeights.append(initializer.initialize(
                parameters.getHiddenNodes(i), parameters.getHiddenNodes(i), random)
            )

        # Call the superclass train method
        super().train(corpus, parameters, initializer)
        
    def clear(self):
        super().clear()
        for l in range(len(self.layers) - 2):
            for m in range(self.aVectors[l].getRow()):
                self.aVectors[l].setValue(m, 0, 0.0)
                self.zVectors[l].setValue(m, 0, 0.0)
                self.rVectors[l].setValue(m, 0, 0.0)

    def calculateOutput(self, sentence, index):
        word = sentence.getWord(index)
        self.createInputVector(word)
        for l in range(len(self.layers) - 2):
            self.rVectors[l].append(self.rWeights[l].multiply(self.layers[l]))
            self.zVectors[l].append(self.zWeights[l].multiply(self.layers[l]))
            self.rVectors[l].append(self.rRecurrentWeights[l].multiply(self.oldLayers[l]))
            self.zVectors[l].append(self.zRecurrentWeights[l].multiply(self.oldLayers[l]))

            self.rVectors[l] = self.activationFunction(self.rVectors[l], self.activationFunction)
            self.zVectors[l] = self.activationFunction(self.zVectors[l], self.activationFunction)

            self.aVectors[l].append(
                self.recurrentWeights[l].multiply(
                    self.rVectors[l].elementProduct(self.oldLayers[l])
                )
            )
            self.aVectors[l].append(self.weights[l].multiply(self.layers[l]))

            self.aVectors[l] = self.activationFunction(self.aVectors[l], ActivationFunction.TANH)

            self.layers[l + 1].append(
                self.calculateOneMinusMatrix(self.zVectors[l]).elementProduct(self.oldLayers[l])
            )
            self.layers[l + 1].append(self.zVectors[l].elementProduct(self.aVectors[l]))
            self.layers[l + 1] = self.biased(self.layers[l + 1])

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

    for l in range(self.parameters.layerSize() - 1, -1, -1):
        delta = deltaWeights[-1].elementProduct(self.zVectors[l]).elementProduct(
            self.derivative(self.aVectors[l], ActivationFunction.TANH)
        )
        zDelta = zDeltaWeights[-1].elementProduct(
            self.aVectors[l].difference(self.oldLayers[l])
        ).elementProduct(
            self.derivative(self.zVectors[l], self.activationFunction)
        )
        rDelta = (
            rDeltaWeights[-1]
            .elementProduct(self.aVectors[l].difference(self.oldLayers[l]))
            .elementProduct(self.derivative(self.zVectors[l], self.activationFunction))
            .transpose()
            .multiply(self.recurrentWeights[l])
            .transpose()
            .elementProduct(self.oldLayers[l])
            .elementProduct(self.derivative(self.rVectors[l], self.activationFunction))
        )

        deltaWeights[-1] = delta.multiply(self.layers[l].transpose())
        deltaRecurrentWeights[-1] = delta.multiply(
            self.rVectors[l].elementProduct(self.oldLayers[l]).transpose()
        )
        zDeltaWeights[-1] = zDelta.multiply(self.layers[l].transpose())
        zDeltaRecurrentWeights[-1] = zDelta.multiply(self.oldLayers[l].transpose())
        rDeltaWeights[-1] = rDelta.multiply(self.layers[l].transpose())
        rDeltaRecurrentWeights[-1] = rDelta.multiply(self.oldLayers[l].transpose())

        if l > 0:
            deltaWeights.append(
                delta.transpose()
                .multiply(
                    self.weights[l].partial(
                        0, self.weights[l].getRow() - 1, 0, self.weights[l].getColumn() - 2
                    )
                )
                .transpose()
            )
            deltaRecurrentWeights.append(deltaWeights[-1].clone())
            zDeltaWeights.append(
                zDelta.transpose()
                .multiply(
                    self.zWeights[l].partial(
                        0, self.zWeights[l].getRow() - 1, 0, self.zWeights[l].getColumn() - 2
                    )
                )
                .transpose()
            )
            zDeltaRecurrentWeights.append(zDeltaWeights[-1].clone())
            rDeltaWeights.append(
                rDelta.transpose()
                .multiply(
                    self.rWeights[l].partial(
                        0, self.rWeights[l].getRow() - 1, 0, self.rWeights[l].getColumn() - 2
                    )
                )
                .transpose()
            )
            rDeltaRecurrentWeights.append(rDeltaWeights[-1].clone())

    self.weights[-1].append(deltaWeights[0])
    deltaWeights.pop(0)

    for l in range(len(deltaWeights)):
        self.weights[-l - 2].append(deltaWeights[l])
        self.rWeights[-l - 1].append(rDeltaWeights[l])
        self.zWeights[-l - 1].append(zDeltaWeights[l])
        self.recurrentWeights[-l - 1].append(deltaRecurrentWeights[l])
        self.zRecurrentWeights[-l - 1].append(zDeltaRecurrentWeights[l])
        self.rRecurrentWeights[-l - 1].append(rDeltaRecurrentWeights[l])
