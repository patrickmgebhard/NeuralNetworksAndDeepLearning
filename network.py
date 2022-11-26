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
    # this loops over all the tuples in the list
    for b, w in zip(self.biases, self.weights):
      # this calculates the final output activation, because it goes through all tuples of weights and biases and calculates the activation for each neuron
      a = sigmoid(np.dot(w, a) + b)
      print(a)
    return a

  # reminder the test_data=None is a default value, if no variable is passed
  # the epoches is the number of times the algorithm goes through all the training data
  # for each epoch
    # we create mini batches from the training data and interate through all mini batches updating the weights and biases after each mini batch
    # if test_data is provided the algorithm is evaluated after each epoch
  def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
    if test_data: n_test = len(test_data)
    n = len(training_data)
    for j in xrange(epochs:
      random.shuffle(training_data)
      # xrange(start, stop, step size)
      mini_batches = [training_data[k:k+mini_batch_size] for k in xrange(0, n, mini_batch_size)]
      for mini_batch in mini_batches:
        self.update_mini_batch(mini_batch, eta)
      if test_data:
        print "Epoch {0}: {1} / {2}".format(j, self.evalue(test_data), n_test)
      else:
        print "Epoch {0} complete".format(j)

  def update_mini_batch(self, mini_batch, eta):
    # TODO so basically the cost function derivatives in the mini batch are added up and then the average is subtracted/added from the weight/bias
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    for x, y in mini_batch:
        delta_nabla_b, delta_nabla_w = self.backprop(x, y)
        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
    self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
    self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
  


  # give a training sample and return a the gradient for the cost function
  def backprop(self, x, y):
    nabla_b = [np.zeros(b.shape) for b in self.biases]
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    # feedforward why is self.feedforward not use
    activation = x
    activations = [x]
    zs = []
    for b, w in zip(self.biases, self.weights):
      z = np.dot(w, activation) + b
      zs.append(z)
      activation = sigmoid(z)
      activations.append(activation)
    # backward pass
    # not so sure if that syntax will work
    delta = self.cost_derivative(activations[-1, y]) * \ sigmoid_prime(zs[-1]))
    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())
    for l in xrange(2, self.num_layers):
      z = zs[-l]
      sp = sigmoid_prime(z)
      delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
    return (nabla_b, nabla_w)
 

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
