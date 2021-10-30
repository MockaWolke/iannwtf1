import numpy as np
from perceptron import Perceptron

class MLP(Perceptron):
    """ Initializes a Multi Layered Perceptron.
        Inherites from Perceptron.
        Methods:
            - forward_step      Passes inputs through network.
            - backprop_step     Updates network params.
    """

    def __init__(self, input_width = 1, output_width = 1, hidden_width = 4, depth = 1):
        """ input_width = input layer width
            output_width = output layer width
            hidden_width = hidden layer width
            depth = number of hidden layers
        """


        self.perceptrons = np.array(
            [super().__init__(input_width)])                                            # create first hidden layer

        self.hidden_layers = np.array(                                                  # create a layer array for possible
            [self.perceptrons])                                                         # additional hidden layers to come


        if depth > 1:
            
            self.perceptrons = np.append(                                              # assuming that all additional hidden
                self.perceptrons,                                                      # layers have the same width
                [super().__init__(hidden_width) for neuron in hidden_width])
            
            self.hidden_layers = np.append(                                            # create as many additional hidden
                self.hidden_layers,                                                    # layers as passed in the params
                [self.perceptrons for layer in depth])

        
        self.output = np.array(                                                        # create output layer
            [super().__init__(hidden_width) for neuron in output_width])



    def forward_step(self, inputs):
        """ Passes inputs through network.
            inputs = activations from earlier layer neurons
        """

        return super()._forward_step(inputs)                                           # calculate activations



    def backprob_step(self):
        """ Updates network params.
            ...
        """

        pass