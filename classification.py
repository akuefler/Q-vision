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

from loadData import RADIUS, HORIZON

def sampleCATdata(n_examples, size, asgray, categories, get_seqs= True, shuffle= True):
    S = sampleCAT(n_examples, size= size, asgray= asgray, categories= categories, get_seqs= get_seqs)

    X_SEQ = None
    Y_SEQ = []
    H, W = S[0].image.shape
    C = 1

    N = len(S)
    X = np.zeros((N, C, H, W))

    Y = np.zeros(len(S), dtype= int)
    for i, trial in enumerate(S):
        if asgray:
            X[i] = np.expand_dims(trial.image, 0)
        else:
            X[i] = trial.image

        Y[i] = categories.index(trial.category)
        if get_seqs:
            for subj, seq in trial.sequence.items():
                if X_SEQ is None:
                    X_SEQ = np.expand_dims(seq, 0)
                else:
                    X_SEQ = np.concatenate((X_SEQ, np.expand_dims(seq, 0)), axis= 0)


                Y_SEQ.append(Y[i])
        #for subject, traj in trial.trajs.items():
            #for fixation in traj:

    #X_SEQS = np.array(X_SEQS)
    Y_SEQ = np.array(Y_SEQ)

    if shuffle:
        p = np.random.permutation(range(N))
        X = X[p]
        Y = Y[p]

    X_train, X_val, X_test = np.array_split(X, [np.floor(0.5 * N), np.floor(0.75*N)])
    Y_train, Y_val, Y_test = np.array_split(Y, [np.floor(0.5 * N), np.floor(0.75*N)])

    if get_seqs:
        N_SEQ = len(X_SEQ)
        p = np.random.permutation(range(N_SEQ))
        X_SEQ = X_SEQ[p]
        Y_SEQ = Y_SEQ[p]

        X_SEQ_train, X_SEQ_val, X_SEQ_test = np.array_split(X_SEQ, [np.floor(0.5 * N_SEQ), np.floor(0.75*N_SEQ)])
        Y_SEQ_train, Y_SEQ_val, Y_SEQ_test = np.array_split(Y_SEQ, [np.floor(0.5 * N_SEQ), np.floor(0.75*N_SEQ)])

        return X_train, X_val, X_test, Y_train, Y_val, Y_test,\
               X_SEQ_train, X_SEQ_val, X_SEQ_test, Y_SEQ_train, Y_SEQ_val, Y_SEQ_test

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

if True:
    categories= ['Action', 'Indoor', 'Object', 'Affective']
    size = (500, 500)
    N = 100
    X_train, X_val, X_test, Y_train, Y_val, Y_test,\
    X_SEQ_train, X_SEQ_val, X_SEQ_test, Y_SEQ_train, Y_SEQ_val, Y_SEQ_test = sampleCATdata(N, size= size, asgray= True, categories= categories)

    #cnn = ConvBaseline(size[0], size[1], 1, 300, len(categories), batch_size= 10)
    print "Constructing Recurrent CNN. This should take a bit..."

    H = 2*RADIUS
    rcnn = ConvLSTM(H, H, 1, HORIZON, 300, len(categories), batch_size= 10)

    #rcnn.predict(X_SEQS_train[0:10])


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

if False:
    ##Highest CIFAR accuracy so far: ~0.21 w 20 filters, 500 hidden units, 60 epochs
    cnn.train(X_train, Y_train, X_val, Y_val, num_epochs= 60) #Lower learning rate seems to make a considerable diff.

    error = mets.accuracy_score(Y_train, cnn.predict(X_train).argmax(axis = 1))
    print("Accuracy on training set", error)

    error = mets.accuracy_score(Y_test, cnn.predict(X_test).argmax(axis = 1))
    print("Accuracy on test set", error)

if True:
    rcnn.train(X_SEQ_train, Y_SEQ_train, X_SEQ_val, Y_SEQ_val, num_epochs= 10)

    error = mets.accuracy_score(Y_SEQ_train, rcnn.predict(X_SEQ_train).argmax(axis = 1))
    print("Accuracy on training set", error)

    error = mets.accuracy_score(Y_SEQ_test, rcnn.predict(X_SEQ_test).argmax(axis = 1))
    print("Accuracy on test set", error)

halt= True