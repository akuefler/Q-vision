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

    X_SEQS = None
    Y_SEQS = []
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
                if X_SEQS is None:
                    X_SEQS = np.expand_dims(seq, 0)
                else:
                    X_SEQS = np.concatenate((X_SEQS, np.expand_dims(seq, 0)), axis= 0)


                Y_SEQS.append(Y[i])
        #for subject, traj in trial.trajs.items():
            #for fixation in traj:

    #X_SEQS = np.array(X_SEQS)
    Y_SEQS = np.array(Y_SEQS)

    if shuffle:
        p = np.random.permutation(range(N))
        X = X[p]
        Y = Y[p]

    X_train, X_val, X_test = np.array_split(X, [np.floor(0.5 * N), np.floor(0.75*N)])
    Y_train, Y_val, Y_test = np.array_split(Y, [np.floor(0.5 * N), np.floor(0.75*N)])

    if get_seqs:
        N_SEQS = len(X_SEQS)
        p = np.random.permutation(range(N_SEQS))
        X_SEQS = X_SEQS[p]
        Y_SEQS = Y_SEQS[p]

        X_SEQS_train, X_SEQS_val, X_SEQS_test = np.array_split(X_SEQS, [np.floor(0.5 * N_SEQS), np.floor(0.75*N_SEQS)])
        Y_SEQS_train, Y_SEQS_val, Y_SEQS_test = np.array_split(Y_SEQS, [np.floor(0.5 * N_SEQS), np.floor(0.75*N_SEQS)])

        return X_train, X_val, X_test, Y_train, Y_val, Y_test,\
               X_SEQS_train, X_SEQS_val, X_SEQS_test, Y_SEQS_train, Y_SEQS_val, Y_SEQS_test

    return X_train, X_val, X_test, Y_train, Y_val, Y_test

if True:
    categories= ['Action', 'Indoor', 'Object', 'Affective']
    size = (500, 500)
    N = 20
    X_train, X_val, X_test, Y_train, Y_val, Y_test,\
    X_SEQS_train, X_SEQS_val, X_SEQS_test, Y_SEQS_train, Y_SEQS_val, Y_SEQS_test = sampleCATdata(N, size= size, asgray= True, categories= categories)

    #cnn = ConvBaseline(size[0], size[1], 1, 300, len(categories), batch_size= 10)
    print "Constructing Recurrent CNN. This should take a bit..."

    H = 2*RADIUS
    rcnn = ConvLSTM(H, H, 1, HORIZON, 300, len(categories), batch_size= 10)

    rcnn.predict(X_SEQS_train[0:10])


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

##Highest CIFAR accuracy so far: ~0.21 w 20 filters, 500 hidden units, 60 epochs
cnn.train(X_train, Y_train, X_val, Y_val, num_epochs= 60) #Lower learning rate seems to make a considerable diff.

error = mets.accuracy_score(Y_train, cnn.predict(X_train).argmax(axis = 1))
print("Accuracy on training set", error)

error = mets.accuracy_score(Y_test, cnn.predict(X_test).argmax(axis = 1))
print("Accuracy on test set", error)

halt= True