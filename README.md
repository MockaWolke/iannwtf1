# Implementing Artificial Neural Networks with Tensorflow - Group 2

## Weeks

1.  [Backpropagation from Scratch](backprop_from_scratch)

This week our task was to implement a MLP from scratch. We did so and used different classes and skripts for our Dataset, Functions, Perceptron, training and a Jupyter Notebook for visualization. We also visualized every possible logical gate as well as a live training visualization of our network.

## dataset.py Script
Provides the inputs with the labels. Can be choosen between and, or, nand. nor, xor depending on the given parameter (standard is xor).

## eval.py Script
Contains methods to calculate sigmoid, sigmoidprime, squarred error and accuracy.

## perceptron.py Class
Represents a single instance of one Perceptron with methods to calculate a forward step with activateion funciton sigmoid as well as an update method to update the parameters of the respective Perceptron instance. Perceptron is initialized with a learning rate of 1, activation function of sigmoid and activation functions derivative of sigmoidprime. Other activation function or lerning rate could be choosen if needed.

## mlp.py Class 
Represents our MLP. The constructor initializes our mlp and there are additional methods for passing the inputs through the network and another method to update the parameters. The MLP is initilized with 2 input units, 1 output neuron and 4 neurons in 2 hidden layers. The global lerning rate is 1 and the activateion function is sigmoid again. These parameters can be changed (e.g. more hidden layers or more neurons in it) if necessary. 

## training.py Script
This script contains functions to train our MLP. There is an epoch function wich passes the input through our network as well as a training function (which is also used for visualization. Also there is a function used for our live training in the visualization.py script.

## visualization.py Notebook
This notebook is used to visualize the training and output of our network. We visualize different logical gates as well as a live training of our network. 
