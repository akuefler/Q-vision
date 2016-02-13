import numpy as np
from matplotlib import pyplot as plt

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

        #updates = lasagne.updates.nesterov_momentum(
        #    loss, params, learning_rate= 1e-7, momentum= 0.9)
        updates = lasagne.updates.sgd(loss, params, learning_rate= 1e-5)

        test_prediction = lasagne.layers.get_output(self.network, deterministic= True)
        test_loss = lasagne.objectives.squared_error(test_prediction, self.y)
        test_loss = test_loss.mean()
        #test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), self.y),
        #                  dtype=theano.config.floatX)

        ##Methods:
        self.weight_update = theano.function(inputs= [self.x, self.y], outputs= loss, updates= updates)
        #self.weight_update = theano.function(inputs= [self.x, self.y], outputs= self.loss(self.x, self.y), updates= updates)
        self.validate = theano.function([self.x, self.y], [test_loss])
        self.predict = theano.function([self.x], lasagne.layers.get_output(self.network))

    def train(self, X_train, y_train, X_val, y_val, num_epochs):
        #input_var = self.x
        #target_var = self.y

        #prediction = lasagne.layers.get_output(self.network, inputs = input_var)
        #loss = lasagne.objectives.squared_error(prediction, target_var)
        #loss = loss.mean()

        #params = lasagne.layers.get_all_params(self.network, trainable=True)
        #updates = lasagne.updates.sgd(loss, params, learning_rate= 1e-10)

        #train_fn = theano.function([input_var, target_var], loss, updates=updates)

        for epoch in range(num_epochs):
            train_loss = 0.0
            train_batches = 0.0

            losses = []

            for batch in iterate_minibatches(X_train, y_train, self.batch_size):
                inputs, targets = batch
                train_loss += self.weight_update(inputs, targets)
                #train_loss += train_fn(inputs, targets)
                losses.append(train_loss)
                train_batches += 1

            #plt.plot(losses)

            val_err = 0.0
            val_batches = 0.0
            for batch in iterate_minibatches(X_val, y_val, self.batch_size):
                inputs, targets = batch
                err = self.validate(inputs, targets)
                val_err += err[0]
                #val_acc += acc
                val_batches += 1

            print("  training loss:\t\t{:.6f}".format(train_loss / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            #print("  validation accuracy:\t\t{:.2f} %".format(
            #    val_acc / val_batches * 100))

class SigNet(NeuralNet):
    def __init__(self, n_features, hidden_size, output_size, batch_size, use_batch_norm= False):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        l_in = lasagne.layers.InputLayer(shape= (batch_size, n_features),
                                         input_var= self.x)

        #l_hid1 = lasagne.layers.DenseLayer(
            #l_in, num_units= hidden_size,
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(gain= 'relu')
            #)
        l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform()
            )

        l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.GlorotUniform())

        #Might optimize this by performing nonlinearity and affine transform in one layer.
        l_out = lasagne.layers.NonlinearityLayer(l_hid2, nonlinearity= lasagne.nonlinearities.sigmoid)

        self.layers = [l_in, l_hid1, l_hid2, l_out]
        self.network = l_out

        self.initiliaze()


class EuclidNet(NeuralNet):
    def __init__(self, n_features, hidden_size, output_size, batch_size):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        l_in = lasagne.layers.InputLayer(shape= (batch_size, n_features),
                                         input_var= self.x)

        #l_hid1 = lasagne.layers.DenseLayer(
            #l_in, num_units= hidden_size,
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(gain= 'relu'))

        ##Pretty sure tanh is going to help keep neurons in right range.
        l_hid1 = lasagne.layers.DenseLayer(
            l_in, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())

        l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear)

        self.layers = [l_in, l_hid1, l_out]
        self.network = l_out

        self.initiliaze()