import numpy as np
from matplotlib import pyplot as plt

import theano
import theano.tensor as T

import lasagne

from collections import OrderedDict

import sklearn as sk
from sklearn import metrics as mets

UNTIL_LOSS = 0.7

def iterate_minibatches(inputs, targets, batchsize, shuffle= True):
    if shuffle:
        p = np.random.permutation(range(inputs.shape[0]))
        inputs = inputs[p]
        targets = targets[p]

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
        self.x1 = T.matrix('inputs')
        self.x2 = T.matrix('inputs')
        self.y = T.matrix('targets')
        self.network = None

    def initiliaze(self, mode= 'regress'):
        """
        Create the rest of the attributes/methods after a computational graph has been created.
        """
        #Training parameters:
        if mode == 'regress':
            loss = lasagne.objectives.squared_error(lasagne.layers.get_output(self.network, deterministic= False), self.y)
            loss = loss.mean() + self.penalty

            test_prediction = lasagne.layers.get_output(self.network, deterministic= True)
            test_loss = lasagne.objectives.squared_error(test_prediction, self.y)
            test_loss = test_loss.mean() + self.penalty
        elif mode == 'classify':
            loss = lasagne.objectives.multiclass_hinge_loss(lasagne.layers.get_output(self.network), self.y)
            #loss = lasagne.objectives.multiclass_hinge_loss(lasagne.layers.get_output(self.network, deterministic= False), self.y)
            loss = loss.mean() + self.penalty

            test_prediction = lasagne.layers.get_output(self.network, deterministic= True)
            test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction, self.y)
            test_loss = test_loss.mean() + self.penalty
        elif mode == 'sequence':
            ##Only consider last label of the sequence?
            loss = self.objective(lasagne.layers.get_output(self.network, deterministic= False), self.y)
            loss = loss.mean() + self.penalty

            test_prediction = self.objective(self.network, deterministic= True)
            test_loss = lasagne.objectives.multiclass_hinge_loss(test_prediction, self.y)
            test_loss = test_loss.mean() + self.penalty

        else:
            assert False

        params = lasagne.layers.get_all_params(self.network, trainable=True)

        #updates = lasagne.updates.nesterov_momentum(
        #    loss, params, learning_rate= 1e-7, momentum= 0.9)
        #updates = lasagne.updates.sgd(loss, params, learning_rate= 1e-5)
        updates = lasagne.updates.adam(loss, params, learning_rate= 1e-4)
        #updates = lasagne.updates.adam(loss, params, learning_rate= 8e-4)

        ##Methods:
        if type(self) == SplitNet \
           or type(self) == ConvSplitNet \
           or type(self) == MultiConvSplitNet \
           or type(self) == SigNet \
           or type(self) == EuclidNet:
            self.weight_update = theano.function(inputs= [self.x1, self.x2, self.y], outputs= loss, updates= updates)
            #self.weight_update = theano.function(inputs= [self.x, self.y], outputs= self.loss(self.x, self.y), updates= updates)
            self.validate = theano.function([self.x1, self.x2, self.y], [test_loss])
            self.predict = theano.function([self.x1, self.x2], lasagne.layers.get_output(self.network))
        else:
            self.weight_update = theano.function(inputs= [self.x1, self.y], outputs= loss, updates= updates)
            self.validate = theano.function([self.x1, self.y], [test_loss])
            self.predict = theano.function([self.x1], lasagne.layers.get_output(self.network))


    def print_accuracy(self, X, Y):
        N = X.shape[0]
        batch_size = 200
        accs = []
        for i in range(0, N, batch_size):
            #print("batch: ", i)
            x = X[i:i+batch_size]
            y = Y[i:i+batch_size]

            Xout = self.predict(x)
            Xpred = Xout.argmax(axis = 1)
            acc = mets.accuracy_score(Xpred, y)

            accs.append(acc)

        accs = np.array(accs)
        print("Accuracy is: ", accs.mean())

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

            #val_err = 0.0
            #val_batches = 0.0
            #for batch in iterate_minibatches(X_val, y_val, self.batch_size):
                #inputs, targets = batch
                #err = self.validate(inputs, targets)
                #val_err += err[0]
                ##val_acc += acc
                #val_batches += 1

            print("EPOCH: {:,.1f}".format(epoch))
            print("  training loss:\t\t{:.6f}".format(train_loss / train_batches))
            if (epoch+1) % 10 == 0:
                print "Train Accuracy: "
                self.print_accuracy(X_train, y_train)
                print "Validation Accuracy: "
                self.print_accuracy(X_val, y_val)

            #print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))

            #if val_err / val_batches < UNTIL_LOSS:
                #print "Stopping early..."
                #break
            #print("  validation accuracy:\t\t{:.2f} %".format(
            #    val_acc / val_batches * 100))

class SigNet(NeuralNet):
    def __init__(self, n_features1, n_features2, hidden_size, output_size, batch_size, use_batch_norm= False, reg= 0.0):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        l_in1 = lasagne.layers.InputLayer(shape= (batch_size, n_features1),
                                         input_var= self.x1)

        l_in2 = lasagne.layers.InputLayer(shape= (batch_size, n_features2),
                                          input_var= self.x2)

        l_cat = lasagne.layers.ConcatLayer([l_in1, l_in2])

        #l_hid1 = lasagne.layers.DenseLayer(
            #l_in, num_units= hidden_size,
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(gain= 'relu')
            #)
        l_hid1 = lasagne.layers.DenseLayer(
            l_cat, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform()
            )

        l_hid2 = lasagne.layers.DenseLayer(
            l_hid1, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear,
            W=lasagne.init.GlorotUniform())

        #Might optimize this by performing nonlinearity and affine transform in one layer.
        l_out = lasagne.layers.NonlinearityLayer(l_hid2, nonlinearity= lasagne.nonlinearities.sigmoid)

        self.layers = [l_in1, l_in2, l_cat, l_hid1, l_hid2, l_out]
        self.network = l_out

        self.penalty = lasagne.regularization.regularize_network_params(self.layers, penalty= lasagne.regularization.l2) * reg

        self.initiliaze()

class SplitNet(NeuralNet):
    def __init__(self, n_features1, n_features2, hidden_size, output_size, batch_size, use_batch_norm= False, reg= 0.0):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        #Stream 1
        l_in1 = lasagne.layers.InputLayer(shape= (batch_size, n_features1),
                                         input_var= self.x1)


        l_hid1 = lasagne.layers.DenseLayer(
            l_in1, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotUniform()
            )

        #Stream 2
        l_in2 = lasagne.layers.InputLayer(shape= (batch_size, n_features2),
                                          input_var= self.x2)

        l_hid2 = lasagne.layers.DenseLayer(
            l_in2, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotUniform()
        )

        l_cat = lasagne.layers.ConcatLayer([l_hid1, l_hid2])

        #Tanh layer to keep values in correct range.
        l_hid3 = lasagne.layers.DenseLayer(
            l_cat, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())

        #Might optimize this by performing nonlinearity and affine transform in one layer.
        l_out = lasagne.layers.DenseLayer(
            l_hid3, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.sigmoid,
            W=lasagne.init.GlorotUniform())

        self.layers = [l_in1, l_hid1, l_in2, l_hid2, l_cat, l_hid3, l_out]
        self.network = l_out

        self.penalty = lasagne.regularization.regularize_network_params(self.layers, penalty= lasagne.regularization.l2) * reg

        self.initiliaze()

class ConvSplitNet(NeuralNet):
    def __init__(self, height, width, channels, n_features2,
                 hidden_size, output_size, batch_size,
                 output= 'sig',
                 num_filters= 10, pool_param = 4,
                 use_batch_norm= False, reg= 0.0):
        NeuralNet.__init__(self)
        self.batch_size = batch_size
        self.layers = []

        self.x1 = T.tensor4('inputs')
        #self.x2 = T.tensor4('targets')
        #self.y = T.dvector('targets')

        ##Stream 1
        l_in1 = lasagne.layers.InputLayer(shape= (None, channels, height, width),
                                         input_var= self.x1)
        self.layers.append(l_in1)

        l_conv = lasagne.layers.Conv2DLayer(
            l_in1, num_filters= num_filters,
            filter_size=3, stride= 1, pad=1,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotUniform())
        self.layers.append(l_conv)

        l_pool = lasagne.layers.MaxPool2DLayer(l_conv, pool_size= pool_param, stride= pool_param)

        l_reshape = lasagne.layers.ReshapeLayer(l_pool, (-1, np.prod(l_pool.output_shape[1:])))

        ##Stream 2
        l_in2 = lasagne.layers.InputLayer(shape= (None, n_features2),
                                          input_var= self.x2)
        self.layers.append(l_in2)

        l_hid2 = lasagne.layers.DenseLayer(
            l_in2, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotUniform()
        )
        self.layers.append(l_hid2)

        l_cat = lasagne.layers.ConcatLayer([l_reshape, l_hid2])
        self.layers.append(l_cat)

        #Tanh layer to keep values in correct range.
        l_hid3 = lasagne.layers.DenseLayer(
            l_cat, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())
        self.layers.append(l_hid3)

        if output == 'sig':
            l_out = lasagne.layers.DenseLayer(
                l_hid3, num_units= output_size,
                nonlinearity=lasagne.nonlinearities.sigmoid,
                W=lasagne.init.GlorotUniform())
            self.layers.append(l_out)

        elif output == 'euclid':
            l_out = lasagne.layers.DenseLayer(
                l_hid3, num_units= output_size,
                nonlinearity=lasagne.nonlinearities.linear,
                W=lasagne.init.GlorotUniform())
            self.layers.append(l_out)

        self.network = l_out

        self.penalty = lasagne.regularization.regularize_network_params(self.layers, penalty= lasagne.regularization.l2) * reg

        self.initiliaze(mode= 'regress')


class MultiConvSplitNet(NeuralNet):
    def __init__(self, height, width, channels, n_features2,
                 hidden_size, output_size, batch_size,
                 output= 'sig',
                 num_filters= 10,
                 use_batch_norm= False, reg= 0.0):
        NeuralNet.__init__(self)
        self.batch_size = batch_size
        self.layers = []

        self.x1 = T.tensor4('inputs')
        self.x2 = T.tensor4('targets')
        self.y = T.ivector('targets')

        ##Stream 1
        l_in1 = lasagne.layers.InputLayer(shape= (batch_size, channels, height, width),
                                         input_var= self.x1)
        self.layers.append(l_in1)


        l_conv1 = lasagne.layers.Conv2DLayer(
            l_in1, num_filters= num_filters,
            filter_size=3, stride= 1, pad=1,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotUniform())
        self.layers.append(l_conv1)

        #Downsample with larger stride.
        l_conv2 = lasagne.layers.Conv2DLayer(
            l_conv1, num_filters= num_filters,
            filter_size=3, stride= 2,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotUniform())
        self.layers.append(l_conv2)

        l_conv3 = lasagne.layers.Conv2DLayer(
            l_conv2, num_filters= num_filters,
            filter_size=3, stride= 3,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotUniform())
        self.layers.append(l_conv3)

        l_reshape = lasagne.layers.ReshapeLayer(l_conv3, (-1, np.prod(l_conv3.output_shape[1:])))

        ##Stream 2
        l_in2 = lasagne.layers.InputLayer(shape= (batch_size, n_features2),
                                          input_var= self.x2)
        self.layers.append(l_in2)

        l_hid2 = lasagne.layers.DenseLayer(
            l_in2, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.elu,
            W=lasagne.init.GlorotUniform()
        )
        self.layers.append(l_hid2)

        l_cat = lasagne.layers.ConcatLayer([l_hid1, l_hid2])
        self.layers.append(l_cat)

        #Tanh layer to keep values in correct range.
        l_hid3 = lasagne.layers.DenseLayer(
            l_cat, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())
        self.layers.append(l_hid3)

        if output == 'sig':
            l_out = lasagne.layers.DenseLayer(
                l_hid3, num_units= output_size,
                nonlinearity=lasagne.nonlinearities.sigmoid,
                W=lasagne.init.GlorotUniform())
            self.layers.append(l_out)

        elif output == 'euclid':
            l_out = lasagne.layers.DenseLayer(
                l_hid3, num_units= output_size,
                nonlinearity=lasagne.nonlinearities.linear,
                W=lasagne.init.GlorotUniform())
            self.layers.append(l_out)

        self.network = l_out

        self.penalty = lasagne.regularization.regularize_network_params(self.layers, penalty= lasagne.regularization.l2) * reg

        self.initiliaze()

class EuclidNet(NeuralNet):
    def __init__(self, n_features1, n_features2, hidden_size, output_size, batch_size, reg= 0.0):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        l_in1 = lasagne.layers.InputLayer(shape= (batch_size, n_features1),
                                          input_var= self.x1)

        l_in2 = lasagne.layers.InputLayer(shape= (batch_size, n_features2),
                                          input_var= self.x2)

        l_cat = lasagne.layers.ConcatLayer([l_in1, l_in2])

        #l_in = lasagne.layers.InputLayer(shape= (batch_size, n_features),
                                         #input_var= self.x1)

        #l_hid1 = lasagne.layers.DenseLayer(
            #l_in, num_units= hidden_size,
            #nonlinearity=lasagne.nonlinearities.rectify,
            #W=lasagne.init.HeUniform(gain= 'relu'))

        ##Pretty sure tanh is going to help keep neurons in right range.
        l_hid1 = lasagne.layers.DenseLayer(
            l_cat, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.tanh,
            W=lasagne.init.GlorotUniform())

        l_out = lasagne.layers.DenseLayer(
            l_hid1, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.linear)

        self.layers = [l_in1, l_in2, l_cat, l_hid1, l_out]
        self.network = l_out

        self.penalty = lasagne.regularization.regularize_network_params(self.layers, penalty= lasagne.regularization.l2) * reg

        self.initiliaze()

class ConvBaseline(NeuralNet):
    def __init__(self, height, width, channels, hidden_size, output_size, batch_size,
                 pre_conv= False, num_convpools= 2, num_hiddense= 2, pool_param= 4, num_filters= 5, reg= 1e-2):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        self.x1 = T.tensor4('inputs')
        #self.y = T.lvector('targets')
        self.y = T.ivector('targets')
        self.layers = []

        l_in = lasagne.layers.InputLayer(shape= (None, channels, height, width),
                                         input_var= self.x1)
        self.layers.append(l_in)

        if pre_conv == True:
            pre_conv = lasagne.layers.Conv2DLayer(
                l_in, num_filters= num_filters,
                filter_size=3, stride= 1, pad=1,
                nonlinearity=lasagne.nonlinearities.elu,
                W=lasagne.init.GlorotUniform())
            self.layers.append(pre_conv)

            prev_layer = pre_conv
        else:
            prev_layer = l_in

        for cp in range(num_convpools):
            conv = lasagne.layers.Conv2DLayer(
                prev_layer, num_filters= num_filters,
                filter_size=3, stride= 1, pad=1,
                nonlinearity=lasagne.nonlinearities.elu,
                W=lasagne.init.GlorotUniform())
            self.layers.append(conv)

            pool = lasagne.layers.MaxPool2DLayer(conv, pool_size= pool_param, stride= pool_param)
            self.layers.append(pool)

            prev_layer = pool

        drop = lasagne.layers.DropoutLayer(prev_layer)
        prev_layer = drop
        for hd in range(num_hiddense):
            dense = lasagne.layers.Conv2DLayer(
                prev_layer, num_filters= num_filters,
                filter_size=3, stride= 1, pad=1,
                nonlinearity=lasagne.nonlinearities.elu,
                W=lasagne.init.GlorotUniform())
            self.layers.append(dense)

            prev_layer = dense

        l_out = lasagne.layers.DenseLayer(
            dense, num_units= output_size,
            nonlinearity=lasagne.nonlinearities.softmax)
        self.layers.append(l_out)

        self.network = l_out

        self.penalty = lasagne.regularization.regularize_network_params(self.layers, penalty= lasagne.regularization.l2) * reg

        self.initiliaze(mode= 'classify')


class ConvRNN(NeuralNet):
    def __init__(self, height, width, channels, timesteps, hidden_size, output_size, batch_size, num_convpools= 2, num_filters= 5):
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        tensor5 = T.TensorType('float64', [False]*5)
        self.x1 = tensor5('inputs')
        self.y = T.lvector('targets')
        self.layers = []

        n_batch, n_steps, n_channels, width, height = (batch_size, timesteps, channels, width, height)
        n_out_filters = 7
        filter_shape = (3, 3)

        l_in = lasagne.layers.InputLayer(
             (None, n_steps, n_channels, width, height),
             input_var= self.x1)
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
    def __init__(self, height, width, channels, timesteps, hidden_size, output_size, batch_size,
                 num_convpools= 2, num_filters= 5, reg= 0.0,
                 objective= lasagne.objectives.multiclass_hinge_loss):
        """
        max pooling works well with softmax/cross entropy loss. mean works better for hinge.

        Hinge/Global pool seems better overall. Should be able to achieve val accuracy ~33%. Hinge can overfit data.
        Cross entropy doesn't seem able to overfit training set.

        softmax + slice_out --> 0.36 val acc. Then it starts overfitting and val acc drops to chance.

        multihinge + slice_out --> 0.39 val acc. Got up to .37, dropped to 0.25 during overfitting. Then jumped again.
        """
        NeuralNet.__init__(self)
        self.batch_size = batch_size

        tensor5 = T.TensorType('float64', [False]*5)
        self.x1 = tensor5('inputs')
        self.y = T.lvector('targets')
        self.layers = []

        n_batch, n_steps, n_channels, width, height = (batch_size, timesteps, channels, width, height)
        n_out_filters = 7
        filter_shape = (3, 3)

        l_in = lasagne.layers.InputLayer(
             (None, n_steps, n_channels, width, height),
             input_var= self.x1)
        self.layers.append(l_in)

        l_reshape1 = lasagne.layers.ReshapeLayer(l_in, (-1, l_in.output_shape[2], l_in.output_shape[3], l_in.output_shape[4]))
        self.layers.append(l_reshape1)

        conv = lasagne.layers.Conv2DLayer(
            l_reshape1,
            n_out_filters, filter_shape,
            nonlinearity=lasagne.nonlinearities.elu, pad='same')
        self.layers.append(conv)

        pool = lasagne.layers.MaxPool2DLayer(conv, pool_size=4, stride=4)
        self.layers.append(pool)

        linear = lasagne.layers.DenseLayer(pool, num_units= hidden_size, nonlinearity= lasagne.nonlinearities.elu) #This was linear.
        self.layers.append(linear)

        l_reshape2 = lasagne.layers.ReshapeLayer(linear, (-1, timesteps, hidden_size))
        self.layers.append(l_reshape2)

        lstm = lasagne.layers.LSTMLayer(l_reshape2, hidden_size)
        self.layers.append(lstm)

        dense1 = lasagne.layers.DenseLayer(
            pool, num_units= hidden_size,
            nonlinearity=lasagne.nonlinearities.elu, #Rectify may be causing nans in cross entropy loss.
            W=lasagne.init.GlorotUniform())
        self.layers.append(dense1)

        if objective == lasagne.objectives.multiclass_hinge_loss:
            dense2 = lasagne.layers.DenseLayer(
                dense1, num_units= output_size,
                nonlinearity=lasagne.nonlinearities.linear)
            #self.layers.append(dense2)
        elif objective == lasagne.objectives.categorical_crossentropy:
            dense2 = lasagne.layers.DenseLayer(
                dense1, num_units= output_size,
                nonlinearity=lasagne.nonlinearities.softmax)
            #self.layers.append(dense2)
        else:
            assert False
        self.layers.append(dense2)

        #Compute argmax of all timesteps?
        l_reshape3 = lasagne.layers.ReshapeLayer(dense2, (-1, output_size, timesteps))
        self.layers.append(l_reshape3)

        l_out = lasagne.layers.SliceLayer(l_reshape3, indices=-1, axis=2)
        #l_out = lasagne.layers.GlobalPoolLayer(l_reshape3, pool_function=T.max) #Using mean instead of max seems to improve val. accuracy.
        self.layers.append(l_out)
        #l_out = dense2

        #self.layers = [l_in, l_rec, l_out]
        self.network = l_out

        self.penalty = lasagne.regularization.regularize_network_params(self.layers, penalty= lasagne.regularization.l2) * reg
        self.objective = objective

        self.initiliaze(mode= 'sequence')
