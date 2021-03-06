# Implementing Artificial Neural Networks with Tensorflow - Group 2

<details>
  
  <summary>
    <h2>HW1 - course_prep<h2>
  </summary>

  All files are located in [course_prep](course_prep).

  Recap of some python operations and calculus.

</details>

<details>
  
  <summary>
    <h2> HW2 - backprop_from_scratch <h2>
  </summary>

  All files are located in [backprop_from_scratch](backprop_from_scratch).

  This week our task was to implement a MLP from scratch. We did so and used different classes and skripts for our Dataset, Functions, Perceptron, training and a Jupyter Notebook for visualization.

  We have created a quite sofisticated fully connected MLP. You can chose as many input, output neurons, and hidden layers as you want as well as adjusting the hiddenlayer width freely.
  You can even specify different activation functions for the input, hidden and output neurons.
  We also visualized every possible logical gate as well as a live training visualization of our network.

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

<details>
  
  <summary>
    <h2>HW3 - mlp_bacteria_classification<h2>
  </summary>

  All files are located in [HW3_bacteria](HW3_bacteria).

  This week our task was to work with Tensorflow datasets and to create a Newtwork working with some data about different kind of bacteria which can be differantiated by their respective genome sequence. For that matter the data had to be preprocessed with one-hot encoding. After that we created a Layer and Model class to realize the different layers in our network and the network itself. For the training an visualization we used a jupyter notebook for easier use and easier visualization.

  ### [preprocessing.py](HW3_bacteria/preprocessing.py)
  This skript is used to preprocess our data. First we defined a function onehotify which one-hot encodes our dataset. We then used this function in our prep_data function to apply the one-hot encoding to our tensorflow dataset.

  ### [layer.py](HW3_bacteria/layer.py)
  This class represents a Layer in our network. The constructor initializes a layer (default is with 256 units and sigmoid activation function). The build function  creates random weights and bias and the call function is used to update our parameters.

  ### [model.py](HW3_bacteria/model.py)
  This class represents our whole model. It is initialized with 2 hidden layers and an output layer. The call function is used to call the call function of our layer class to update our parameters.

  ### [HW3.ipynb](HW3_bacteria/HW3.ipynb)
  This jupyter notebook is used for our visualization and training. It is inbedded in Google colab so you do not have to run it locally. First this reposetory is cloned so the respective classes and skripts can be utilized. After that we defined a function to train our model and to test our model. After that the data is loaded, preprocessed, hyperparameters are choosen, test is initialized, performance is printed and then the model is trained. After that the visualization follows.

</details>

<details>

  <summary>
    <h2>HW4 - mlp_wine_quality_classification<h2>
  </summary>

</details>

<details>

  <summary>
    <h2>HW5 - cnn_mnist_classification<h2>
  </summary>

</details>

<details>

  <summary>
    <h2>HW6 - resNet_denseNet_cifar10<h2>
  </summary>

</details>

<details>

  <summary>
    <h2>HW7 - lstm_from_scratch<h2>
  </summary>

</details>

<details>

  <summary>
    <h2>HW8 - autoencoder_mnist<h2>
  </summary>

</details>

<details>

  <summary>
    <h2>HW9 - gan_quickDraw<h2>
  </summary>

</details>

<details>

  <summary>
    <h2>HW10 - skipGram_bible<h2>
  </summary>

</details>

<details>

  <summary>
    <h2>HW11 - transformer_nietzsche<h2>
  </summary>

</details>