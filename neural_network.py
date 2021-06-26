import numpy as np

def sigmoid(x):
    # sigmoid aktivasyon fonksiyonu
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    # ???
    return x * (1 - x)


class NeuralNetwork:

    def __init__(self):
        np.random.seed(1)
        # başlangıç ağırlıkları veriliyor
        self.synaptic_weights = 2 * np.random.random((11, 1)) - 1

    def train(self, training_inputs, training_outputs, training_iterations):
        # iteration sayısı kadar eğitiyor. Her adımda ağırlıklar (snyaptic_weights) güncelleniyor
        for iteration in range(training_iterations):
            # çıktı alınıyor
            output = self.think(training_inputs)
            # hata hesaplanıyor
           # error=((training_outputs-output)**2)
            error = ((training_outputs-output)**2)/11
            #error = training_outputs - output
            # buradaki işlem her proje için farklı, bir kural yok
            adjustments = np.dot(training_inputs.T, error * sigmoid_derivative(output))
            # ağırlık matrisi güncelleniyor
            self.synaptic_weights += adjustments

    def think(self, inputs):
        # hidden layer yok, en basit formül ile ağırlık*input = output
        inputs = inputs.astype(float)
        output = sigmoid(np.dot(inputs, self.synaptic_weights))
        return output