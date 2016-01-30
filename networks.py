import numpy as np

import theano
import theano.tensor as T

import lasagne

from collections import OrderedDict

def iterate_minibatches(inputs, targets, batchsize):
    mini_batches = []
    if batchsize == targets.shape[0]:
        mini_batches.append((inputs, targets))
        return mini_batches

    for i, x in enumerate(inputs):
        input_batch = inputs[i*batchsize:i*batchsize + batchsize, :]
        target_batch = targets[i*batchsize:i*batchsize + batchsize, :]

        if input_batch.size == 0 or target_batch.size == 0:
            break

        mini_batches.append((input_batch, target_batch))

    return mini_batches

class NeuralNet(object):
    def __init__(self):
        self.x = T.matrix('inputs')
        self.y = T.matrix('targets')
        self.network = None

    def initiliaze(self):
        """
        Create the rest of the attributes/methods after a computational graph has been created.
        """
        #Training parameters:
        loss = lasagne.objectives.squared_error(lasagne.layers.get_output(self.network), self.y)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate= 0.01, momentum= 0.9)

        #Methods:
        self.weight_update = theano.function(inputs= [self.x, self.y], outputs= loss, updates= updates)
        self.predict = theano.function([self.x], lasagne.layers.get_output(self.network))

class SigNet(NeuralNet):
    def __init__(self, n_features, hidden_size, output_size, batch_size, use_batch_norm= False):
        NeuralNet.__init__(self)

        l_in = lasagne.layers.InputLayer(shape= (batch_size, n_features),
                                         input_var= self.x)

        l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear)

        #Might optimize this by performing nonlinearity and affine transform in one layer.
        l_out = lasagne.layers.NonlinearityLayer(l_hid2, nonlinearity= lasagne.nonlinearities.sigmoid)

        self.layers = [l_in, l_hid1, l_hid2, l_out]
        self.network = l_out

        self.initiliaze()


class EuclidNet(NeuralNet):
    def __init__(self, n_features, hidden_size, output_size, batch_size):
        NeuralNet.__init__(self)

        l_in = lasagne.layers.InputLayer(shape= (batch_size, n_features),
                                         input_var= self.x)

        l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())

        l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear)

        self.layers = [l_in, l_hid1, l_out]
        self.network = l_out

        self.initiliaze()
