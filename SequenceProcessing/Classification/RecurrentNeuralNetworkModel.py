from SequenceProcessing.Classification.Model import Model
from Corpus.Sentence import Sentence


class RecurrentNeuralNetworkModel(Model):
    def calculateOutput(self, sentence: Sentence, index: int):
        word = sentence.getWord(index)  # Assume it returns a LabelledVectorizedWord
        self.createInputVector(word)

        for l in range(len(self.layers) - 2):
            # Add recurrent weights multiplied by the old layer values
            self.layers[l + 1].add(self.recurrentWeights[l].multiply(self.oldLayers[l]))
            # Add weights multiplied by the current layer values
            self.layers[l + 1].add(self.weights[l].multiply(self.layers[l]))
            # Apply activation function and bias
            self.layers[l + 1] = self.calculateActivationFunction(self.layers[l + 1], self.activationFunction)
            self.layers[l + 1] = self.biased(self.layers[l + 1])

        # Process the final output layer
        self.layers[len(self.layers) - 1].add(
            self.weights[len(self.weights) - 1].multiply(self.layers[len(self.layers) - 2])
        )
        self.normalizeOutput()

    def backpropagation(self, sentence: Sentence, index: int, learningRate: float):
        word = sentence.getWord(index)  # Assume it returns a LabelledVectorizedWord
        rMinusY = self.calculateRMinusY(word)
        rMinusY.multiplyWithConstant(learningRate)

        deltaWeights = []
        deltaRecurrentWeights = []

        # Calculate delta for the output layer
        deltaWeights.append(rMinusY.multiply(self.layers[len(self.layers) - 2].transpose()))
        deltaWeights.append(rMinusY)
        deltaRecurrentWeights.append(rMinusY)

        # Backpropagation through hidden layers
        for l in range(self.parameters.layerSize() - 1, -1, -1):
            partialWeights = self.weights[l + 1].partial(
                0, self.weights[l + 1].getRow() - 1,
                0, self.weights[l + 1].getColumn() - 2
            )
            partialLayers = self.layers[l + 1].partial(
                0, self.layers[l + 1].getRow() - 2,
                0, self.layers[l + 1].getColumn() - 1
            )
            derivativeMatrix = self.derivative(partialLayers, self.activationFunction).transpose()
            delta = (deltaWeights[len(deltaWeights) - 1].transpose()
                     .multiply(partialWeights)
                     .elementProduct(derivativeMatrix)
                     .transpose())

            deltaWeights[len(deltaWeights) - 1] = delta.multiply(self.layers[l].transpose())
            deltaRecurrentWeights[len(deltaRecurrentWeights) - 1] = delta.multiply(self.oldLayers[l].transpose())

            if l > 0:
                deltaWeights.append(delta)
                deltaRecurrentWeights.append(delta)

        # Update weights
        self.weights[len(self.weights) - 1].add(deltaWeights.pop(0)) # pop returns 0th element and removes it
        for l in range(len(deltaWeights)):
            self.weights[len(self.weights) - l - 2].add(deltaWeights[l])
            self.recurrentWeights[len(self.recurrentWeights) - l - 1].add(deltaRecurrentWeights[l])
