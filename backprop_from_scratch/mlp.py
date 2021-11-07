"""
Class MLP

Represents our Multilayer Perceptron. Initializes a Multilayer Perceptron and contains functions to calculate forwards as well as backward (backpropagation) steps.
"""
import numpy as np
from perceptron import Perceptron
from eval import sigmoidprime , sigmoid
import types

class MLP(Perceptron):
    """ Initializes a Multi Layered Perceptron.
        Inherites from Perceptron.
        Methods:
            -   forward_step      Passes inputs through network.
            -   backprop_step     Updates network params.
    """

    def __init__(self, input_units = 2, output_width = 1, hidden_width = 4, depth = 2, activ_funcs = sigmoid, func_derivates = sigmoidprime, global_learning_rate = 1):
        """ Constructor
            ## Parameters:
        
            -   input_units = input layer width
            -   output_width = output layer width
            -   hidden_width = hidden layer width
            -   depth = number of hidden layers
            -   activ_funcs = can either be one single activ func or a list of three -> (activ_func_for_the_first_layer,hidden_layers,output_layers)
            -   activ_func_derivates = derivatives of activ funcs, must match format of the activ_func
            -   global_learning_rate = global learning rate 
        """
        
        # Here we just check wether the activ funcs have the wright format
        assert isinstance(activ_funcs,types.FunctionType) or len(activ_funcs) == 3, "Not the correct format for the activ funcs" 
        assert (isinstance(activ_funcs,types.FunctionType) and isinstance(func_derivates,types.FunctionType)) or len(activ_func) == 3, "Not the correct activ func derivation does not match activ func" 
        if isinstance(activ_funcs,types.FunctionType): # here we just transform the activ functions to a list, so that we have them in a nice format for the creation of our layers
            self.activ_funcs=[activ_funcs for i in range(3)]
            self.activ_func_derivates = [func_derivates for i in range(3)]
        else:
            self.activ_funcs = activ_funcs
            self.activ_func_derivates = func_derivates
            
        self.global_learning_rate = global_learning_rate

        self.hidden_width = hidden_width 
        self.depth = depth
        self.input_units = input_units
        self.output_width = output_width

        # create array and fill it up with on single Perceptron so that we have the correct datatype for our array
        self.hidden_layers = np.full((self.hidden_width,depth),Perceptron(1))
        
        # changing that all the entries of our hidden_layers array corespond to one single Perceptron, with excatcly the same random values
        # iterating through hidden layer
        for y in range(self.hidden_width): 
            for x in range(self.depth) :
                # the input layer has a different number of incoming weights than other hidden layers ('input_units')
                # so we have to create different perceptrons for the first layer
                if x == 0:
                    self.hidden_layers[y,x] = Perceptron(self.input_units,learning_rate= global_learning_rate, activ_func = self.activ_funcs[0], activ_func_derivate = self.activ_func_derivates[0] )
                # else there are hidden_with amount of incomming neurons 
                else:
                    self.hidden_layers[y,x] = Perceptron(self.hidden_width,learning_rate= global_learning_rate, activ_func = self.activ_funcs[1], activ_func_derivate = self.activ_func_derivates[1])
        
        # the output layer has a different width and needs to be stored seperately
        self.output_layer = np.array([Perceptron(self.hidden_width, learning_rate= global_learning_rate, activ_func = self.activ_funcs[2], activ_func_derivate = self.activ_func_derivates[2]) for i in range(self.output_width)])


    def forward_step(self, inputs):
        """ Passes inputs through network.

            ## Parameters:

            - inputs = activations from earlier layer neurons
        """
        
        inputs_for_next_layer = inputs
        
        # storage of predictions
        self.prediction = np.nan 

        # compute the forward step for every perceptron in our hidden_layer, again iterate through our whole hidden layer
        for x in range(self.depth):
            layer =  self.hidden_layers[:,x]
            inputs_for_next_layer = [neuron._forward_step(inputs_for_next_layer) for neuron in layer]
                
        # now for our output layer
        self.prediction = [neuron._forward_step(inputs_for_next_layer) for neuron in self.output_layer]
            
        return self.prediction


    def backprob_step(self, t):
        """ Updates network params.

            ## Parameters:
            
            - inputs = activations from earlier layer neurons
            - t = target output
        """
        
        # now the backprop step of our output layer first
        for neuron in self.output_layer: 
            delta  = (2/self.output_width) *(neuron.output- t) * neuron.activation_func_der(neuron.drive)   # here we use the formula for the respective deltas         
            neuron._update(delta)    # and update the neurons
    
        # Let's now for the last layer of our hidden layer
        layer =  self.hidden_layers[:,-1] # let's get the last layer
        deltas_from_output_layer = [float(output_neuron.delta) for output_neuron in self.output_layer] # and a list of the deltas

        for y, neuron in enumerate(layer):
            # we have already saved the deltas of our last layer in deltas_for_next_layer, no we only need the coresponding weights
            coresponding_weights = [output_neuron.weights[y] for output_neuron in self.output_layer] # compute the coresponding weights for the delta
            delta = np.dot(deltas_from_output_layer, coresponding_weights) * neuron.activation_func_der(neuron.drive) # compute the delta nicely with np.dot
            neuron._update(delta)
            
        for x in range(self.depth-2,-1,-1):
            layer =  self.hidden_layers[:,x] # get this layer
            next_layer = self.hidden_layers[:,x+1] # and get the behind our current layer
            
            deltas_from_previous_layer = [float(neuron_of_next_layer.delta) for neuron_of_next_layer in next_layer] # get the deltas of the next layer 
            for y, neuron in enumerate(layer):
                coresponding_weights = [neuron_of_next_layer.weights[y] for neuron_of_next_layer in next_layer]
                delta = np.dot(deltas_from_previous_layer,coresponding_weights) * neuron.activation_func_der(neuron.drive)  # compute the delta nicely with np.dot
                neuron._update(delta)   # last but not least update