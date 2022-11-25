# testing how to commit and push stuff back to GitHub
# TODO really understand these numpy arrays and what they represent

import random
import numpy as np

class Network(object):
  
  def __init__(self, sizes):
    self.num_layers = len(sizes)
    self.sizes = sizes
    # this creates a two-dimensional numpy array with the initialized biases that are normally distributed
    self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
    # zip() takes two iterables e.g. lists and then returns a list of tuples with the elements
    # again the weights are initialized with a normal distribution
    self.weights = [np.random.rand(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

  def feedforward(self, a):
    # TODO understand this for loop!
    for b, w in zip(self.biases, self.weights):
      # this calculates the final output activation, because it goes through all tuples of weights and biases and calculates the activation for each neuron
      a = sigmoid(np.dot(w, a) + b)
      print(a)
    return a

  #reminder the test_data=None is a default value, if no variable is passed
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    print("yo momma")

  # this function returns the no. of correctly guessed examples
  def evaluate(self, test_data):
    # the np.argmax gives the index of the maximum value in the output array
    test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
    # int(True) = 1 and int(False) = 0 
    return sum(int(x == y) for (x, y) in test_results)
        
  # TODO understand as I don't get it      
  def cost_derivative(self, output_activations, y):
    return (output_activations-y)

# the sigmoid function
def sigmoid(z):
  return 1.0/(1.0+np.exp(-z))

# the derivative of the sigmoid function
def sigmoid_prime(z):
  return sigmoid(z)*(1-sigmoid(z))


# Playing around with the class and its methods

net = Network([1, 1, 1])
activ = sigmoid(-0.1)

#print(net.num_layers)
#print(net.sizes)
"""print("Weights: ")
print(net.weights)
print("Biases: ")
print(net.biases)
"""
print("zip of biases and weights: ")
print(zip(net.biases, net.weights))
print("Results of feedforward method: ")
net.feedforward([1])
#print(sigmoid(net.weights[0] * 1 + net.biases[0]))

#print(np.random.randn(10))

#print(activ)
