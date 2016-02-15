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
        ##ISSUE: This is all probably totally unnecessary...
        #if Dx == 3:
            #input_batch = inputs[i*batchsize:i*batchsize + batchsize, :]
            ##target_batch = targets[i*batchsize:i*batchsize + batchsize, :]
        #elif Dx == 4:
            #input_batch = inputs[i*batchsize:i*batchsize + batchsize, :, :]
            ##target_batch = targets[i*batchsize:i*batchsize + batchsize, :]

        #elif Dx == 5:
            #assert False
        input_batch = inputs[i*batchsize:i*batchsize + batchsize, :, :]

        #if Dy == 2:
            #target_batch = targets[i*batchsize:i*batchsize + batchsize, :]
        #elif Dy == 1:
            #target_batch = targets[i*batchsize:i*batchsize + batchsize]
        #else:
            #assert False
        target_batch = targets[i*batchsize:i*batchsize + batchsize]

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
        elif mode == 'sequence':
            ##Only consider last label of the sequence?
            loss = lasagne.objectives.multiclass_hinge_loss(lasagne.layers.get_output(self.network), self.y)
            loss = loss.mean()

            test_prediction = lasagne.layers.get_output(self.network, deterministic= True)
            test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction, self.y)
            test_loss = test_loss.mean()

        else:
            assert False

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        #updates = lasagne.updates.nesterov_momentum(
        #    loss, params, learning_rate= 1e-7, momentum= 0.9)
        #updates = lasagne.updates.sgd(loss, params, learning_rate= 1e-5)
        updates = lasagne.updates.adam(loss, params, learning_rate= 1e-5)

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


class ConvRNN(NeuralNet):
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
             (None, n_steps, n_channels, width, height),
             input_var= self.x)
        l_in_to_hid = lasagne.layers.Conv2DLayer(
             lasagne.layers.InputLayer((None, n_channels, width, height)),
             n_out_filters, filter_shape, pad='same')
        l_hid_to_hid = lasagne.layers.Conv2DLayer(
             lasagne.layers.InputLayer(l_in_to_hid.output_shape),
             n_out_filters, filter_shape, pad='same')
        l_rec = lasagne.layers.CustomRecurrentLayer(
             l_in, l_in_to_hid, l_hid_to_hid)

        l_reshape = lasagne.layers.ReshapeLayer(l_rec, (-1, np.prod(l_rec.output_shape[2:])))

        l_out = lasagne.layers.DenseLayer(
            l_reshape, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear)

        self.layers = [l_in, l_rec, l_out]
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
             (None, n_steps, n_channels, width, height),
             input_var= self.x)
        l_reshape1 = lasagne.layers.ReshapeLayer(l_in, (-1, l_in.output_shape[2], l_in.output_shape[3], l_in.output_shape[4]))
        conv = lasagne.layers.Conv2DLayer(
            l_reshape1,
            n_out_filters, filter_shape, pad='same')
        pool = lasagne.layers.MaxPool2DLayer(conv, pool_size=4, stride=4)
        linear = lasagne.layers.DenseLayer(pool, num_units= hidden_size, nonlinearity= lasagne.nonlinearities.linear)
        l_reshape2 = lasagne.layers.ReshapeLayer(linear, (-1, timesteps, hidden_size))
        lstm = lasagne.layers.LSTMLayer(l_reshape2, hidden_size)

        dense1 = lasagne.layers.DenseLayer(
            pool, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh, #Rectify may be causing nans in cross entropy loss.
            W=lasagne.init.GlorotUniform())

        dense2 = lasagne.layers.DenseLayer(
            dense1, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear)

        #Compute argmax of all timesteps?
        l_reshape3 = lasagne.layers.ReshapeLayer(dense2, (-1, output_size, timesteps))
        l_out = lasagne.layers.GlobalPoolLayer(l_reshape3, pool_function=T.mean) #Using mean instead of max seems to improve val. accuracy.
        #l_out = dense2

        #self.layers = [l_in, l_rec, l_out]
        self.network = l_out

        self.initiliaze(mode= 'sequence')

#class ConvLSTM(NeuralNet):
    #def __init__(self, height, width, channels, timesteps, hidden_size, output_size, batch_size, num_convpools= 2, num_filters= 5):
        #NeuralNet.__init__(self)
        #self.batch_size = batch_size

        #tensor5 = T.TensorType('float64', [False]*5)
        #self.x = tensor5('inputs')
        #self.y = T.lvector('targets')
        #self.layers = []

        #n_batch, n_steps, n_channels, width, height = (batch_size, timesteps, channels, width, height)
        #n_out_filters = 7
        #filter_shape = (3, 3)

        #l_in = lasagne.layers.InputLayer(
             #(None, n_steps, n_channels, width, height),
             #input_var= self.x)
        #l_in_to_hid = ConvPoolLayer(
             #lasagne.layers.InputLayer((None, n_channels, width, height)),
             #n_out_filters, filter_shape, pad='same')
        ##l_hid_to_hid = lasagne.layers.Conv2DLayer(
        ##     lasagne.layers.InputLayer(l_in_to_hid.output_shape),
        ##     n_out_filters, filter_shape, pad='same')
        #l_hid_to_hid = DenseLSTMLayer(
            #lasagne.layers.InputLayer(l_in_to_hid.output_shape),
            #num_units= 500)
        ##l_hid_to_hid.input_shape = l_hid_to_hid.input_shapes[0] #ISSUE: SEEMS VERY UNSAFE.

        #l_rec = lasagne.layers.CustomRecurrentLayer(
             #l_in, l_in_to_hid, l_hid_to_hid)

        #l_reshape = lasagne.layers.ReshapeLayer(l_rec, (-1, np.prod(l_rec.output_shape[2:])))

        #l_out = lasagne.layers.DenseLayer(
            #l_reshape, num_units= output_size,
            #nonlinearity=lasagne.nonlinearities.linear)

        #self.layers = [l_in, l_rec, l_out]
        #self.network = l_out

        ##Convolutions
        ##prev_layer = lasagne.layers.InputLayer((None, n_channels, width, height))
        ##for cp in range(num_convpools):
            ##conv = lasagne.layers.Conv2DLayer(
                ##incoming = prev_layer,
                ##num_filters= num_filters,
                ##filter_size=3, stride= 1, pad=1,
                ##nonlinearity=lasagne.nonlinearities.tanh,
                ##W=lasagne.init.GlorotUniform())
            ##self.layers.append(conv)

            ##pool = lasagne.layers.MaxPool2DLayer(conv, pool_size=4, stride=4)
            ###self.layers.append(pool)

            ##prev_layer = pool

        ##l_hid = lasagne.layers.DenseLayer(
            ##pool, num_units= hidden_size,
            ##nonlinearity=lasagne.nonlinearities.tanh, #Rectify may be causing nans in cross entropy loss.
            ##W=lasagne.init.GlorotUniform())
        ###self.layers.append(l_hid3)

        ##l_lstm = lasagne.layers.LSTMLayer(l_hid, num_units= hidden_size)
        ##l_out = lasagne.layers.DenseLayer(
            ##l_lstm, num_units= output_size,
            ##nonlinearity=lasagne.nonlinearities.linear)

        #self.network = l_out
        #self.initiliaze(mode= 'classify')

        ##Recurrence
        ##l_in_to_hid = lasagne.layers.Conv2DLayer(
             ##lasagne.layers.InputLayer((None, n_channels, width, height)),
             ##n_out_filters, filter_shape, pad='same')
        ##l_hid_to_hid = lasagne.layers.Conv2DLayer(
             ##lasagne.layers.InputLayer(l_in_to_hid.output_shape),
             ##n_out_filters, filter_shape, pad='same')
        ##l_rec = lasagne.layers.CustomRecurrentLayer(
             ##l_in, l_in_to_hid, l_hid_to_hid)

        ##l_reshape = lasagne.layers.ReshapeLayer(l_rec, (-1, np.prod(l_rec.output_shape[2:])))

        ##l_out = lasagne.layers.DenseLayer(
            ##l_reshape, num_units= output_size,
            ##nonlinearity=lasagne.nonlinearities.linear)

        ##self.layers = [l_in, l_rec, l_out]
        ##self.network = l_out

        ##self.initiliaze(mode= 'classify')

class DenseLSTMLayer(lasagne.layers.Layer):
    def __init__(self, incoming, num_units, name=None, **kwargs):
        #lasagne.layers.Layer.__init__(self)
        super(DenseLSTMLayer, self).__init__(incoming, **kwargs)
        (batch_size, in_units) = incoming.output_shape

        dense = lasagne.layers.DenseLayer(
            lasagne.layers.InputLayer((None, in_units)), num_units= 500,
            nonlinearity=lasagne.nonlinearities.tanh, #Rectify may be causing nans in cross entropy loss.
            W=lasagne.init.GlorotUniform())

        lstm = lasagne.layers.LSTMLayer(dense, num_units= 500)
        self.num_units= 500

        self.only_return_final = lstm.only_return_final
        self.mask_incoming_index = lstm.mask_incoming_index
        self.hid_init_incoming_index = lstm.hid_init_incoming_index
        self.cell_init_incoming_index = lstm.cell_init_incoming_index
        self.W_in_to_ingate = lstm.W_in_to_ingate
        self.W_in_to_forgetgate = lstm.W_in_to_forgetgate
        self.W_in_to_cell = lstm.W_in_to_cell
        self.W_in_to_outgate = lstm.W_in_to_outgate
        self.W_hid_to_ingate = lstm.W_hid_to_ingate
        self.W_hid_to_forgetgate = lstm.W_hid_to_forgetgate
        self.W_hid_to_cell = lstm.W_hid_to_cell
        self.W_hid_to_outgate = lstm.W_hid_to_outgate
        self.b_ingate = lstm.b_ingate
        self.b_forgetgate = lstm.b_forgetgate
        self.b_cell = lstm.b_cell
        self.b_outgate = lstm.b_outgate
        self.precompute_input = lstm.precompute_input

        self.input_shapes = lstm.input_shapes

    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_shapes)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        if self.only_return_final:
            return input_shape[0], self.num_units
        # Otherwise, the shape will be (n_batch, n_steps, num_units)
        else:
            return input_shape[0], input_shape[1], self.num_units

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Treat all dimensions after the second as flattened feature dimensions
        if input.ndim > 3:
            input = T.flatten(input, 3)

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        input = input.dimshuffle(1, 0, 2)
        seq_len, num_batch, _ = input.shape

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=1)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=1)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (n_time_steps, n_batch, 4*num_units).
            input = T.dot(input, W_in_stacked) + b_stacked

        # At each call to scan, input_n will be (n_time_steps, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.dot(input_n, W_in_stacked) + b_stacked

            # Calculate gates pre-activations and slice
            gates = input_n + T.dot(hid_previous, W_hid_stacked)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]


class ConvPoolLayer(lasagne.layers.Layer):
    def __init__(self, incoming, n_out_filters, filter_shape, pad='same', name=None, **kwargs):
        #lasagne.layers.Layer.__init__(self)
        super(ConvPoolLayer, self).__init__(incoming, **kwargs)
        (batch_size, n_channels, width, height) = incoming.output_shape

        conv = lasagne.layers.Conv2DLayer(
            lasagne.layers.InputLayer((None, n_channels, width, height)),
            n_out_filters, filter_shape, pad='same')

        pool = lasagne.layers.MaxPool2DLayer(conv, pool_size=4, stride=4)

        self.num_units = 500
        dense = lasagne.layers.DenseLayer(
            pool, num_units= self.num_units,
            nonlinearity=lasagne.nonlinearities.tanh, #Rectify may be causing nans in cross entropy loss.
            W=lasagne.init.GlorotUniform())

        self.W = dense.W
        self.b = dense.b
        self.nonlinearity = dense.nonlinearity


    @property
    def output_shape(self):
        return self.get_output_shape_for(self.input_shape)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_units)

    def get_output_for(self, input, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)

        activation = T.dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)






