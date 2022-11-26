# load the MINIST dataset here
import cPickle 
import gzip

import numpy as np

# input are the images
def load_data():
    f = gzip.open('./data/mnist.pkl.gz', 'rb')
    # how does the cPickle know how to separate the data
    training_data, validation_data, test_data = cPickle.load(f)
    f.close()
    #return (training_data[0].shape, training_data[0][0], training_data[1].shape, training_data[1][0], validation_data[0].shape, validation_data[1].shape, test_data[0].shape, test_data[1].shape,)
    return(training_data, validation_data, test_data)

def load_data_wrapper_pg():
    tr_d, va_d, te_d = load_data()
    # training data should be a list containing 50,000 2-tuples (x, y) where x is a 784-dimensional numpy.ndarray with the input image and y is a 10-dimensional numpy.ndarray representing the unit vector corresponding to the correct digit for x
    #print(tr_d, len(tr_d), tr_d[0].shape, tr_d[1].shape)
    # this is already in the correct form
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]] 
    # bring y in correct form
    training_results = [vectorized_result(y) for y in tr_d[1]] 
    # bring x & y together using zip()
    training_data = zip(training_inputs, training_results)
    # validation and test data are lists containing 10,000 2-tuples (x, y) where x is a 784-dimensional array and y is the integer value of the corresponding x
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    test_inputs = [np.reshape(x, (784,1 )) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

print(load_data_wrapper_pg())

# output are the images split into training, test and validation dataset    