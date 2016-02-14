from loadData import *
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.decomposition as deco

import cPickle

from sklearn import metrics as mets
import scipy
import scipy.io
import scipy.ndimage

from networks import *

import math

def sampleCATdata(n_examples, size, asgray, categories):
    S = sampleCAT(n_examples, size= size, asgray= asgray, categories= categories)

    H, W = S[0].image.shape
    C = 1

    cnn = ConvBaseline(H, W, 1, 300, len(categories), batch_size= 10)

    N = len(S)
    X = np.zeros((N, C, H, W))
    Y = np.zeros(len(S), dtype= int)
    for i, trial in enumerate(S):
        X[i, 0] = trial.image
        Y[i] = categories.index(trial.category)

    if shuffle:
        p = np.random.permutation(range(N))
        X = X[p]
        Y = Y[p]

    X_train, X_val, X_test = np.array_split(X, 3)
    Y_train, Y_val, Y_test = np.array_split(Y, 3)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

if False:
    categories= ['Action', 'Indoor', 'Object', 'Affective']
    X_train, X_val, X_test, Y_train, Y_val, Y_test = sampleCATdata(100, size= (500, 500), asgray= True, categories= categories)
else:
    fo = open('./data/cifar-10-batches-py/data_batch_1', 'rb')
    dic = cPickle.load(fo)
    fo.close()
    asgray= False

    X = dic['data']
    Y = dic['labels']

    X_list = np.array_split(X, 3)
    for i, x in enumerate(X_list):
        N, D = x.shape
        IMG = x.reshape((N, 3, 32, 32))
        C = 3
        if asgray:
            C = 1
            IMG = np.transpose(IMG, [0, 3, 2, 1])
            IMG = mpl.colors.rgb_to_hsv(IMG)[:,:,:,2]
            IMG = np.expand_dims(IMG, 1)

        if i == 0: #Train
            X_train = IMG
        elif i == 1: #Val
            X_val = IMG
        elif i == 2: #Test
            X_test = IMG

    Y_train, Y_val, Y_test = np.array_split(Y, 3)
    cnn = ConvBaseline(32, 32, C, 500, 10, batch_size= 200, num_filters= 20)

##Highest accuracy so far: ~0.21 w 20 filters, 500 hidden units, 60 epochs
cnn.train(X_train, Y_train, X_val, Y_val, num_epochs= 60) #Lower learning rate seems to make a considerable diff.

error = mets.accuracy_score(Y_train, cnn.predict(X_train).argmax(axis = 1))
print("Accuracy on training set", error)

error = mets.accuracy_score(Y_test, cnn.predict(X_test).argmax(axis = 1))
print("Accuracy on test set", error)

halt= True