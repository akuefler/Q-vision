import numpy as np
from matplotlib import pyplot as plt

import theano
import theano.tensor as T

import lasagne

from collections import OrderedDict

def iterate_minibatches(inputs, targets, batchsize):
    mini_batches = []
    Dx = inputs.ndim
    Dy = targets.ndim
    if batchsize == targets.shape[0]:
        mini_batches.append((inputs, targets))
        return mini_batches

    for i, x in enumerate(inputs):
        if Dx == 3:
            input_batch = inputs[i*batchsize:i*batchsize + batchsize, :]
            #target_batch = targets[i*batchsize:i*batchsize + batchsize, :]
        elif Dx == 4:
            input_batch = inputs[i*batchsize:i*batchsize + batchsize, :, :]
            #target_batch = targets[i*batchsize:i*batchsize + batchsize, :]

        else:
            assert False

        if Dy == 2:
            target_batch = targets[i*batchsize:i*batchsize + batchsize, :]
        elif Dy == 1:
            target_batch = targets[i*batchsize:i*batchsize + batchsize]
        else:
            assert False

        if input_batch.size == 0 or target_batch.size == 0:
            break

        #if input_batch.shape[0] == batchsize: #ISSUE: Just drop anything the wrong size.
        mini_batches.append((input_batch, target_batch))

    return mini_batches

class NeuralNet(object):
    def __init__(self):
        self.x = T.matrix('inputs')
        self.y = T.matrix('targets')
        self.network = None

    def initiliaze(self, mode= 'regress'):
        """
        Create the rest of the attributes/methods after a computational graph has been created.
        """
        #Training parameters:
        if mode == 'regress':
            loss = lasagne.objectives.squared_error(lasagne.layers.get_output(self.network), self.y)
            loss = loss.mean()

            test_prediction = lasagne.layers.get_output(self.network, deterministic= True)
            test_loss = lasagne.objectives.squared_error(test_prediction, self.y)
            test_loss = test_loss.mean()
        elif mode == 'classify':
            #loss = lasagne.objectives.categorical_crossentropy(lasagne.layers.get_output(self.network), self.y)
            loss = lasagne.objectives.multiclass_hinge_loss(lasagne.layers.get_output(self.network), self.y)
            loss = loss.mean()

            test_prediction = lasagne.layers.get_output(self.network, deterministic= True)
            test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction, self.y)
            test_loss = test_loss.mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        #updates = lasagne.updates.nesterov_momentum(
        #    loss, params, learning_rate= 1e-7, momentum= 0.9)
        #updates = lasagne.updates.sgd(loss, params, learning_rate= 1e-5)
        updates = lasagne.updates.adam(loss, params, learning_rate= 1e-5)

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

            plt.plot(losses)

            val_err = 0.0
            val_batches = 0.0
            for batch in iterate_minibatches(X_val, y_val, self.batch_size):
                inputs, targets = batch
                err = self.validate(inputs, targets)
                val_err += err[0]
                #val_acc += acc
                val_batches += 1

            print("EPOCH: {:,.1f}".format(epoch))
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

class ConvBaseline(NeuralNet):
    def __init__(self, height, width, channels, hidden_size, output_size, batch_size, num_convpools= 2, num_filters= 5):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        self.x = T.tensor4('inputs')
        self.y = T.lvector('targets')
        self.layers = []

        l_in = lasagne.layers.InputLayer(shape= (None, channels, height, width),
                                         input_var= self.x)
        self.layers.append(l_in)

        prev_layer = l_in
        for cp in range(num_convpools):
            conv = lasagne.layers.Conv2DLayer(
                prev_layer, num_filters= num_filters,
                filter_size=3, stride= 1, pad=1,
                nonlinearity=lasagne.nonlinearities.tanh,
                W=lasagne.init.GlorotUniform())
            self.layers.append(conv)

            pool = lasagne.layers.MaxPool2DLayer(conv, pool_size=4, stride=4)
            self.layers.append(pool)

            prev_layer = pool

        l_hid3 = lasagne.layers.DenseLayer(
            pool, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh, #Rectify may be causing nans in cross entropy loss.
            W=lasagne.init.GlorotUniform())
        self.layers.append(l_hid3)

        l_out = lasagne.layers.DenseLayer(
            l_hid3, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear)
        self.layers.append(l_out)

        self.network = l_out

        self.initiliaze(mode= 'classify')


class ConvLSTM(NeuralNet):
    def __init__(self, height, width, channels, timesteps, hidden_size, output_size, batch_size, num_convpools= 2, num_filters= 5):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        tensor5 = T.TensorType('float64', [False]*5)
        self.x = tensor5('inputs')
        self.y = T.lvector('targets')
        self.layers = []

        n_batch, n_steps, n_channels, width, height = (batch_size, timesteps, channels, width, height)
        n_out_filters = 7
        filter_shape = (3, 3)

        l_in = lasagne.layers.InputLayer(
             (n_batch, n_steps, n_channels, width, height),
             input_var= self.x)
        l_in_to_hid = lasagne.layers.Conv2DLayer(
             lasagne.layers.InputLayer((None, n_channels, width, height)),
             n_out_filters, filter_shape, pad='same')
        l_hid_to_hid = lasagne.layers.Conv2DLayer(
             lasagne.layers.InputLayer(l_in_to_hid.output_shape),
             n_out_filters, filter_shape, pad='same')
        l_rec = lasagne.layers.CustomRecurrentLayer(
             l_in, l_in_to_hid, l_hid_to_hid)

        l_reshape = lasagne.layers.ReshapeLayer(l_rec, (n_batch * n_steps, np.prod(l_rec.output_shape[2:])))

        l_out = lasagne.layers.DenseLayer(
            l_reshape, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear)

        self.layers = [l_in, l_rec, l_out]
        self.network = l_out

        self.initiliaze(mode= 'classify')