from loadData import *
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.decomposition as deco

import cPickle

import copy

from sklearn import metrics as mets
import scipy
import scipy.io
import scipy.ndimage

from networks import *

import math

from loadData import RADIUS, HORIZON

def exact_sampleCATdata(cat_count, mode, size, asgray, get_seqs= None, shuffle= True, save= False):
    S, seq_bank = exactCAT(cat_count, mode, size= size, asgray= asgray, get_seqs= get_seqs)

    X_SEQ = None
    Y_SEQ = []
    H, W = S[0].image.shape
    C = 1

    N = len(S)
    X = np.zeros((N, C, H, W))

    Y = np.zeros(len(S), dtype= int)
    print "Extracting data..."
    for i, trial in enumerate(S):
        print "Got image: ", i, " of ", len(S)
        if asgray:
            X[i] = np.expand_dims(trial.image, 0)
        else:
            X[i] = trial.image

        #Y[i] = cat_count.keys().index(trial.category)
        #if get_seqs is not None:
            #for subj, seq in trial.sequence.items():
                #if X_SEQ is None:
                    #X_SEQ = np.expand_dims(seq, 0)
                #else:
                    #X_SEQ = np.concatenate((X_SEQ, np.expand_dims(seq, 0)), axis= 0)

                #Y_SEQ.append(Y[i])

    print "Extraction complete."
        #for subject, traj in trial.trajs.items():
            #for fixation in traj:

    #X_SEQS = np.array(X_SEQS)
    X_SEQ = seq_bank.X
    Y_SEQ = np.array(seq_bank.Y)

    print "X_SEQ shape: ", X_SEQ.shape
    print "Y_SEQ shape: ", Y_SEQ.shape

    if shuffle:
        p = np.random.permutation(range(N))
        X = X[p]
        Y = Y[p]

    if get_seqs is not None:
        N_SEQ = len(X_SEQ)
        if shuffle:
            p = np.random.permutation(range(N_SEQ))
            X_SEQ = X_SEQ[p]
            Y_SEQ = Y_SEQ[p]

        if save == True:
            print "saving..."
            np.savez('matrices/X_SEQ-N-'+str(X_SEQ.shape[0])+"-mode-"+mode+'-RADIUS-'+str(RADIUS)+'-seqtype-'+get_seqs, X_SEQ)
            np.savez('matrices/Y_SEQ-N-'+str(Y_SEQ.shape[0])+"-mode-"+mode+'-RADIUS-'+str(RADIUS)+'-seqtype-'+get_seqs, Y_SEQ)

        return X_SEQ, Y_SEQ

    return X, Y

categories = ['Action','Affective','Art','BlackWhite','Cartoon','Fractal','Indoor','Inverted',\
              'Jumbled','LineDrawing','LowResolution','Noisy','Object','OutdoorManMade','OutdoorNatural',
              'Pattern','Random','Satelite','Sketch','Social']
cat_count_train = {key:50 for key in categories}
cat_count_val = {key:25 for key in categories}

if False: ##Load baseline for CAT2000 classification.
    cat_count_train = {key:50 for key in categories}
    cat_count_val = {key:25 for key in categories}
    #cat_count_train = {'Action':10, 'Sketch':10}
    #cat_count_val = {'Action':5, 'Sketch':5}

    output_size = len(cat_count_train)
    size = (256, 256)

    X_train, Y_train = exact_sampleCATdata(cat_count_train, "train", size= size, asgray= True, get_seqs= False)
    X_val, Y_val = exact_sampleCATdata(cat_count_val, "val", size= size, asgray= True, get_seqs= False)

    C = 1
    cnn = ConvBaseline(size[0], size[1], C, 600, output_size, pool_param= 2, pre_conv= True, num_convpools= 1,
                       use_batchnorm= True, num_hiddense= 2, batch_size= 5, num_filters= 28, reg= 2e-1, drop_prob= 0.7)

elif False: ##Load SGC for CAT classification
    #cat_count_train = {key:50 for key in categories}
    #cat_count_val = {key:25 for key in categories}
    cat_count_train = {'Action':15, 'Sketch':15}
    cat_count_val = {'Action':5, 'Sketch':5}

    output_size = len(cat_count_train)
    size = (256, 256)

    ##HUMAN
    X_SEQ_train, Y_SEQ_train = exact_sampleCATdata(cat_count_train, "train", size= size, asgray= True, get_seqs= 'human')
    X_SEQ_val, Y_SEQ_val = exact_sampleCATdata(cat_count_val, "val", size= size, asgray= True, get_seqs= 'human')

    ##RANDOM
    #X_SEQ_train, Y_SEQ_train = exact_sampleCATdata(cat_count_train, "train", size= size, asgray= True, get_seqs= 'random')
    #X_SEQ_val, Y_SEQ_val = exact_sampleCATdata(cat_count_val, "val", size= size, asgray= True, get_seqs= 'random')

    ##SALIENCY
    #X_SEQ_train, Y_SEQ_train = exact_sampleCATdata(cat_count_train, "train", size= size, asgray= True, get_seqs= 'saliency')
    #X_SEQ_val, Y_SEQ_val = exact_sampleCATdata(cat_count_val, "val", size= size, asgray= True, get_seqs= 'saliency')

    H = 2*RADIUS
    #rcnn = ConvLSTM(H, H, 1, HORIZON, 500, output_size, batch_size= 10, reg= 0.0)
    print "Instantiating rcnn..."
    #orcnn = old_ConvLSTM(H, H, 1, HORIZON, 500, output_size, batch_size= 10, reg= 0.0)
    rcnn = ConvLSTM(H, H, 1, HORIZON, 500, output_size, batch_size= 10,
                 pre_conv= True, num_convpools= 1, num_hiddense= 1,
                 num_filters= 7, reg= 0.0, pool_param = 4,
                 objective= lasagne.objectives.multiclass_hinge_loss)

    halt= True
    #rcnn.predict(X_SEQS_train[0:10])


elif False: ##Load Baseline for CIFAR classification
    fo1 = open('./data/cifar-10-batches-py/data_batch_1', 'rb')
    dic1 = cPickle.load(fo1)
    fo1.close()

    fo2 = open('./data/cifar-10-batches-py/data_batch_2', 'rb')
    dic2 = cPickle.load(fo2)
    fo2.close()

    fo3 = open('./data/cifar-10-batches-py/data_batch_3', 'rb')
    dic3 = cPickle.load(fo3)
    fo3.close()

    asgray= False

    X = np.row_stack((dic1['data'], dic2['data'], dic3['data']))
    Y = np.concatenate((dic1['labels'], dic2['labels'], dic3['labels']))

    #X = dic1['data']
    #Y = dic1['labels']

    N, M = X.shape

    p = np.random.permutation(N)
    X = X[p]
    Y = Y[p]

    X_list = np.array_split(X, [0.5*N, 0.75*N])
    Y_train, Y_val, Y_test = np.array_split(Y, [0.5*N, 0.75*N])

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

    #Y_train, Y_val, Y_test = np.array_split(X, [0.5*N, 0.75*N])
    #cnn = ConvBaseline(32, 32, C, 500, 10, batch_size= 200, num_filters= 20)
    #cnn = ConvBaseline(32, 32, C, 600, 10, pool_param= 2, num_convpools= 3, batch_size= 200, num_filters= 30)
    #cnn = ConvBaseline(32, 32, C, 600, 10, pool_param= 2, pre_conv= True, num_convpools= 2, batch_size= 200, num_filters= 30, reg= 1e-1)
    cnn = ConvBaseline(32, 32, C, 750, 10, pool_param= 2, pre_conv= True, num_convpools= 2, use_batchnorm= True, num_hiddense= 3, batch_size= 200, num_filters= 32, reg= 1e-1)

elif True:
    #Just save...

    cat_count_train = {key:50 for key in categories}
    cat_count_val = {key:25 for key in categories}

    #cat_count_train = {key:2 for key in categories}
    #cat_count_val = {key:1 for key in categories}

    output_size = len(cat_count_train)
    size = (256, 256)

    ##HUMAN
    #print "Fetching human data..."
    #X_SEQ_train, Y_SEQ_train = exact_sampleCATdata(cat_count_train, "train", size= size, asgray= True, get_seqs= 'human', save= True)
    #X_SEQ_val, Y_SEQ_val = exact_sampleCATdata(cat_count_val, "val", size= size, asgray= True, get_seqs= 'human', save= True)

    ##SALIENCY
    print "Fetching SALIENCY data..."
    X_SEQ_train_sal, Y_SEQ_train_sal = exact_sampleCATdata(cat_count_train, "train", size= size, asgray= True, get_seqs= 'saliency', save= True)
    X_SEQ_val_sal, Y_SEQ_val_sal = exact_sampleCATdata(cat_count_val, "val", size= size, asgray= True, get_seqs= 'saliency', save= True)

    ##RANDOM
    print "Fetching RANDOM data..."
    X_SEQ_train_rand, Y_SEQ_train_rand = exact_sampleCATdata(cat_count_train, "train", size= size, asgray= True, get_seqs= 'random', save= True)
    X_SEQ_val_rand, Y_SEQ_val_rand = exact_sampleCATdata(cat_count_val, "val", size= size, asgray= True, get_seqs= 'random', save= True)



if False:
    ##Highest CIFAR accuracy so far: ~0.21 w 20 filters, 500 hidden units, 60 epochs
    Y_train = Y_train.astype('int32')
    Y_val = Y_val.astype('int32')

    print "Start training..."
    cnn.train(X_train, Y_train, X_val, Y_val, num_epochs= 300, save= True) #Lower learning rate seems to make a considerable diff.

    #X_train_out = cnn.predict(X_train)
    #X_train_pred = X_train_out.argmax(axis = 1)
    #acc_train = mets.accuracy_score(Y_train, X_train_pred)
    #print("Accuracy on training set", acc_train)

    print("TRAIN SET")
    cnn.print_accuracy(X_train, Y_train)

    print("VAL SET")
    cnn.print_accuracy(X_val, Y_val)

    #X_val_out = cnn.predict(X_val)
    #X_val_pred = X_val_out.argmax(axis = 1)
    #acc_val = mets.accuracy_score(Y_val, X_val_pred)
    #print("Accuracy on validation set", acc_val)

if True:
    M1 = np.load('matrices/human/X_SEQ-N-16550-mode-train-RADIUS-20-seqtype-human.npz')
    M2 = np.load('matrices/human/Y_SEQ-N-16550-mode-train-RADIUS-20-seqtype-human.npz')

    M3 = np.load('matrices/human/X_SEQ-N-7921-mode-val-RADIUS-20-seqtype-human.npz')
    M4 = np.load('matrices/human/Y_SEQ-N-7921-mode-val-RADIUS-20-seqtype-human.npz')

    X_SEQ_train_hum = M1.items()[0][1]
    Y_SEQ_train_hum = M2.items()[0][1]

    X_SEQ_val_hum = M3.items()[0][1]
    Y_SEQ_val_hum = M4.items()[0][1]

    output_size = len(cat_count_train)

    H = 2*RADIUS
    print "Instantiating human rcnn..."
    hum_rcnn = ConvLSTM(H, H, 1, HORIZON, 500, output_size, batch_size= 10,
                    pre_conv= True, num_convpools= 1, num_hiddense= 1,
                    num_filters= 7, reg= 0.0, pool_param = 4,
                    objective= lasagne.objectives.multiclass_hinge_loss)
    print "Instantiating sal rcnn..."
    sal_rcnn = ConvLSTM(H, H, 1, HORIZON, 500, output_size, batch_size= 10,
                        pre_conv= True, num_convpools= 1, num_hiddense= 1,
                        num_filters= 7, reg= 0.0, pool_param = 4,
                        objective= lasagne.objectives.multiclass_hinge_loss)
    print "Instantiating rand rcnn..."
    rand_rcnn = ConvLSTM(H, H, 1, HORIZON, 500, output_size, batch_size= 10,
                        pre_conv= True, num_convpools= 1, num_hiddense= 1,
                        num_filters= 7, reg= 0.0, pool_param = 4,
                        objective= lasagne.objectives.multiclass_hinge_loss)

    print "Training human rcnn..."
    hum_rcnn.train(X_SEQ_train_hum, Y_SEQ_train_hum, X_SEQ_val_hum, Y_SEQ_val_hum, num_epochs= 15, save= True)

    print "Training saliency rcnn..."
    sal_rcnn.train(X_SEQ_train_sal, Y_SEQ_train_sal, X_SEQ_val_sal, Y_SEQ_val_sal, num_epochs= 15, save= True)

    print "Training random rcnn..."
    rand_rcnn.train(X_SEQ_train_rand, Y_SEQ_train_rand, X_SEQ_val_rand, Y_SEQ_val_rand, num_epochs= 15, save= True)


if False:
    print "Training rcnn..."
    rcnn.train(X_SEQ_train, Y_SEQ_train, X_SEQ_val, Y_SEQ_val, num_epochs= 300, save= True)

    print("TRAIN SET")
    ##rcnn.print_accuracy(X_train, Y_train)
    rcnn.print_accuracy(X_SEQ_train, Y_SEQ_train)

    print("VAL SET")
    rcnn.print_accuracy(X_SEQ_val, Y_SEQ_val)

    #error = mets.accuracy_score(Y_SEQ_train, rcnn.predict(X_SEQ_train).argmax(axis = 1))
    #print("Accuracy on training set", error)

    #error = mets.accuracy_score(Y_SEQ_test, rcnn.predict(X_SEQ_test).argmax(axis = 1))
    #print("Accuracy on test set", error)

halt= True