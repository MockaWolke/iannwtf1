import numpy as np
from perceptron import Perceptron

class MLP(Perceptron):
    """ Initializes a Multi Layered Perceptron.
        Inherites from Perceptron.
        Methods:
            -   forward_step      Passes inputs through network.
            -   backprop_step     Updates network params.
    """

    def __init__(self, input_units = 2, output_width = 1, hidden_width = 4, depth = 5):
        """ ## Parameters:
        
            -   input_units = input layer width
            -   output_width = output layer width
            -   hidden_width = hidden layer width
            -   depth = number of hidden layers
        """

        self.hidden_width = hidden_width

        # the input layer has a different number of incoming weights than other hidden layers ('input_units')
        self.input_layer = np.full(hidden_width, Perceptron(input_units))

        # all hidden layers will be stored here
        self.hidden_layers = np.array(self.input_layer)
        
        # if additional hidden layers were specified in the MLP initialization
        for layer in range(1,depth):
            self.hidden_layers = np.append( # we append them to our hidden-layer-array with their
                self.hidden_layers, np.full(hidden_width, Perceptron(hidden_width))) # specified width
        
        # the output layer has a different width and needs to be stored seperately
        self.output_layer = np.full(output_width, Perceptron(hidden_width))
        



    def forward_step(self, inputs):
        """ Passes inputs through network.

            ## Parameters:

            inputs = activations from earlier layer neurons
        """

        self.activations = np.empty(0) # storage of resulting activations/drives
        self.prediction = np.empty(0) # storage of prediction(s)

        for neuron in self.hidden_layers:
            self.activations = np.append( # store the activation of every neuron
                self.activations, neuron._forward_step(inputs)) # forward step from Perceptron class

            if self.activations.size % self.hidden_width == 0: # as soon as one layer is done, the next layer
                inputs = self.activations[-self.hidden_width:] # gets the previous layer's activation as drive
  
        for neuron in self.output_layer:
            self.prediction = np.append( # store the output layer's activation(s) as
                self.prediction, neuron._forward_step(inputs)) # the final prediction(s)

        return self.prediction



    def backprob_step(self):
        """ Updates network params.

            ## Parameters:
            
            inputs = activations from earlier layer neurons
        """

        pass