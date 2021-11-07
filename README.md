# Implementing Artificial Neural Networks with Tensorflow - Group 2


## [HW 1](course_prep)
Course Preparation


## [HW 2](backprop_from_scratch)
<details>
  <summary>Backpropagation from Scratch</summary>

  This week our task was to implement a MLP from scratch. We did so and used different classes and skripts for our Dataset, Functions, Perceptron, training and a Jupyter Notebook for visualization. We also visualized every possible logical gate as well as a live training visualization of our network.
  ### [dataset.py](backprop_from_scratch/dataset.py)
  Provides the inputs with the labels. Can be choosen between and, or, nand. nor, xor depending on the given parameter (standard is xor).
  ### [eval.py](backprop_from_scratch/eval.py)
  Contains methods to calculate sigmoid, sigmoidprime, squarred error and accuracy.
  
  ### [perceptron.py](backprop_from_scratch/perceptron.py)
  Represents a single instance of one Perceptron with methods to calculate a forward step with activateion funciton sigmoid as well as an update method to update the parameters of the respective Perceptron instance. Perceptron is initialized with a learning rate of 1, activation function of sigmoid and activation functions derivative of sigmoidprime. Other activation function or lerning rate could be choosen if needed.
  
  ### [mlp.py](backprop_from_scratch/mlp.py)
  Represents our MLP. The constructor initializes our mlp and there are additional methods for passing the inputs through the network and another method to update the parameters. The MLP is initilized with 2 input units, 1 output neuron and 4 neurons in 2 hidden layers. The global lerning rate is 1 and the activateion function is sigmoid again. These parameters can be changed (e.g. more hidden layers or more neurons in it) if necessary. 
  
  ### [training.py](backprop_from_scratch/training.py)
  This script contains functions to train our MLP. There is an epoch function wich passes the input through our network as well as a training function (which is also used for visualization. Also there is a function used for our live training in [visualization.ipynb](backprop_from_scratch/visualization.ipynb).
  
  ### [visualization.ipynb](backprop_from_scratch/visualization.ipynb)
  This notebook is used to visualize the training and output of our network. We visualize different logical gates as well as a live training of our network.

</details>







## [HW 3]()