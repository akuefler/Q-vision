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

def delta_update(loss_or_grads, params, delta, learning_rate, momentum=0.9):
    #updates = sgd(loss_or_grads, params, learning_rate)
    grads = lasagne.updates.get_or_compute_grads(loss_or_grads, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        #updates[param] = param - learning_rate * deltas * grad #ISSUE: Only assuming * delta works how I want...
        updates[param] = param - learning_rate * delta * grad

    return lasagne.updates.apply_nesterov_momentum(updates, momentum=momentum)

class EuclidNet(object):
    def __init__(self, n_features, hidden_size, output_size, batch_size):
        #input_var = T.matrix('inputs')
        self.x = T.matrix('inputs')
        #if output_size > 1:
        #self.d = T.matrix('deltas')
        self.d = T.scalar('delta')
        self.y = T.matrix('targets')
        #else:
            #self.y = T.dvector('targets') #ISSUE: Better to use d vector?

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

    def predict(self, X):
        forward = theano.function([self.x], lasagne.layers.get_output(self.network, self.x))
        return forward(X)

    def compute_gradient(self, X, y):
        #halt= True
        loss = lasagne.objectives.squared_error(self.predict(X), self.y)
        loss = loss.mean()
        gradient = T.grad(loss, X)

        return gradient


    def train(self, X_train, Y_train, X_val= None, Y_val= None, deltas= None, batch_size= 50, num_epochs= 100, lr = 0.01, moment = 0.9):

        prediction = lasagne.layers.get_output(self.network)
        loss = lasagne.objectives.squared_error(prediction, self.y)
        loss = loss.mean()

        params = lasagne.layers.get_all_params(self.network, trainable=True)
        #updates = lasagne.updates.nesterov_momentum(
            #loss, params, learning_rate= lr, momentum= moment)

        updates = delta_update(
            loss, params, self.d, learning_rate= lr, momentum= moment)

        test_prediction = lasagne.layers.get_output(self.network)
        test_loss = lasagne.objectives.squared_error(test_prediction, self.y)
        test_loss = test_loss.mean()

        #train_fn = theano.function([self.x, self.y], loss, updates= updates)
        train_fn = theano.function(inputs= [self.x, self.d, self.y],
                                   outputs = loss,
                                   updates= updates
                                   #updates = [(W, Elemwise), (b, Elemwise)]
                                   #,givens ={
            #self.x: self.x * delta}
            )

        #train_model = theano.function(
            #inputs=[index],
            #outputs=cost,
            #updates=updates,
            #givens={
                #self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                #self.y: train_set_y[index * batch_size: (index + 1) * batch_size]
            #}
        #)

        val_fn = theano.function([self.x, self.y], test_loss)

        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            for batch in iterate_minibatches(X_train, Y_train, batch_size):
                inputs, targets = batch
                train_err += train_fn(inputs, deltas, targets)
                train_batches += 1

            # And a full pass over the validation data:
            #val_err = 0.0
            #val_batches = 0
            #for batch in iterate_minibatches(X_val, Y_val, batch_size):
                #inputs, targets = batch
                #err = val_fn(inputs, targets)
                #val_err += err
                ##val_acc += acc
                #val_batches += 1



#def train(classifier, x, y, X_train, y_train, learning_rate, batch_size= 600):
    #cost = classifier.least_squares_cost(y)
    #index = T.lscalar()

    #X = theano.shared(X_train)
    #y = theano.shared(y_train)

    #g_W = T.grad(cost=cost, wrt=classifier.W)
    #g_b = T.grad(cost=cost, wrt=classifier.b)

    #updates = [(classifier.W, classifier.W - learning_rate * g_W),
               #(classifier.b, classifier.b - learning_rate * g_b)]

    #train_model = theano.function(
        #inputs=[index],
        #outputs=cost,
        #updates=updates,
        #givens={
            #x: X[index * batch_size: (index + 1) * batch_size],
            #y: y[index * batch_size: (index + 1) * batch_size]})

    #minibatch_avg_cost = train_model(minibatch_index)


#class EuclidLayer(object):

    #def __init__(self, inp, n_in, n_out):
        #""" Initialize the parameters of the logistic regression

        #:type input: theano.tensor.TensorType
        #:param input: symbolic variable that describes the input of the
                      #architecture (one minibatch)

        #:type n_in: int
        #:param n_in: number of input units, the dimension of the space in
                     #which the datapoints lie

        #:type n_out: int
        #:param n_out: number of output units, the dimension of the space in
                      #which the labels lie

        #"""
        ## start-snippet-1
        ## initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        #self.fanin = n_in
        #self.fanout = n_out

        #self.W = theano.shared(value=np.zeros((n_in, n_out),dtype=theano.config.floatX),
            #name='W',
            #borrow=True)

        ## initialize the biases b as a vector of n_out 0s
        #self.b = theano.shared(
            #value=np.ones(
                #(n_out,),
                #dtype=theano.config.floatX
                #),
            #name='b',
            #borrow=True
        #)

        ## symbolic expression for computing the matrix of class-membership
        ## probabilities
        ## Where:
        ## W is a matrix where column-k represent the separation hyperplane for
        ## class-k
        ## x is a matrix where row-j  represents input training sample-j
        ## b is a vector where element-k represent the free parameter of
        ## hyperplane-k
        ## self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        ## symbolic description of how to compute prediction as class whose
        ## probability is maximal
        ## self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        ## end-snippet-1

        #self.y_pred = T.dot(inp, self.W) + self.b

        ## parameters of the model
        #self.params = [self.W, self.b]

        ## keep track of model input
        #self.inp = inp

    #def least_squares_cost(self, y):
        #"""
        #"""
        ##return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        #return 0.5 * T.sum(T.pow(self.y_pred - y, 2))
        ## end-snippet-2

    #def errors(self, y):
        #"""Return a float representing the number of errors in the minibatch
        #over the total number of examples of the minibatch ; zero one
        #loss over the size of the minibatch

        #:type y: theano.tensor.TensorType
        #:param y: corresponds to a vector that gives for each example the
                  #correct label
        #"""

        ## check if y has same dimension of y_pred
        #if y.ndim != self.y_pred.ndim:
            #raise TypeError(
                #'y should have the same shape as self.y_pred',
                #('y', y.type, 'y_pred', self.y_pred.type)
            #)
        ## check if y is of the correct datatype
        ##if y.dtype.startswith('int'):
            ## the T.neq operator returns a vector of 0s and 1s, where 1
            ## represents a mistake in prediction
        #return T.mean(T.pow(self.y_pred - y, 2))
        ##else:
            ##raise NotImplementedError()