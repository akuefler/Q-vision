import os
import numpy as np
from collections import Counter

import matplotlib as mpl
from matplotlib import pyplot as plt

import math

import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters

import sklearn as sk
import scipy
import scipy.io
import scipy.ndimage

#RADIUS = 42 #If images are 256x256 and HORIZON is 9, RADIUS should be 42.
#RADIUS = 20
RADIUS = 20
EPSILON = 45
#MEMORY = 1
MEMORY = 9
HORIZON = 9
R_THRESH = 150

##ISSUE: should do more pre-processing. Subtract mean images. etc.

class Trial():
    def __init__(self, cat, idx, img, fixmap, salmap, trajs, orig_shape, center_fm= True, get_seqs= False, seq_bank= None):
        self.category = cat
        self.index = idx

        #if img.ndim == 2:
            #C = 1 #number of channels
        #else:
            #C = 3

        self.image = img
        self.fixmap = fixmap
        self.salmap = salmap

        if center_fm: #ISSUE: Change this when add in color channels?
            fm = fixmap.astype(float)
            self.fixmap = (fm - fm.mean())/fm.std()
            #mask1 = (fm > 10).astype('float')
            #mask2 = (fm < 30).astype('float')
            #mask = (fm > R_THRESH).astype('float')
            #self.fixmap = fm * mask1 * mask2
            #self.fixmap[self.fixmap == 0] = -1
            #self.fixmap[self.fixmap != -1] = 1

        self.trajs = trajs
        self.sequence = {}
        if img.ndim == 2:
            N, M = orig_shape
            C = 1
        else:
            N, M, C = orig_shape

        sal_N, sal_M = salmap.shape

        #self.imageCenter = np.array([math.floor(img.shape[0]/2), math.floor(img.shape[1]/2)])
        #self.imageCenter = np.array([math.floor(img.shape[1]/2), math.floor(img.shape[0]/2)])

        self.imageCenter = np.array([math.floor(M/2.0), math.floor(N/2.0)])

        for sub, traj in self.trajs.items():
            if get_seqs == 'saliency':
                yx = computeMaxima(self.salmap, 5, 20)

                assert (yx <= 1).all()
                assert (yx >= 0).all()

                picks = np.random.choice(np.arange(len(yx)), size= HORIZON, replace= False)
                self.trajs[sub] = yx[picks]

                self.sequence[sub] = np.empty((traj.shape[0], C, 2*RADIUS, 2*RADIUS))
                for i, fix in enumerate(self.trajs[sub]):
                    window = self.observe(fix)
                    self.sequence[sub][i] = np.expand_dims(window, 0)

                seq_bank.addSeq(self.seq[sub].copy(), self.category)

                halt= True

            else:
                self.trajs[sub] -= np.mean(self.trajs[sub], axis= 0)
                self.trajs[sub] += self.imageCenter

                self.trajs[sub] /= np.array([M, N], dtype= float)

                ##Remove invalids in first pass.
                if np.isnan(traj).any():
                    self.trajs.pop(sub)
                    continue

                traj = np.delete(traj, np.nonzero(traj[:,0] * M < 0), axis= 0)
                traj = np.delete(traj, np.nonzero(traj[:,1] * N < 0), axis= 0)
                traj = np.delete(traj, np.nonzero(traj[:,0] * M >= M), axis= 0)
                traj = np.delete(traj, np.nonzero(traj[:,1] * N >= N), axis= 0)
                self.trajs[sub] = traj

                ##Make sure all trajectories satisfy horizon.
                t_length = traj.shape[0]
                if t_length < HORIZON:
                    self.trajs.pop(sub)
                    continue

                elif t_length > HORIZON:
                    traj = traj[:HORIZON, :]

                self.trajs[sub] = traj

                if get_seqs == 'human':
                    self.sequence[sub] = np.empty((traj.shape[0], C, 2*RADIUS, 2*RADIUS))
                    for i, fix in enumerate(traj):
                        window = self.observe(fix)
                        self.sequence[sub][i] = np.expand_dims(window, 0)

                    seq_bank.addSeq(self.sequence[sub].copy(), self.category)

                elif get_seqs == 'random':
                    self.sequence[sub] = np.empty((traj.shape[0], C, 2*RADIUS, 2*RADIUS))
                    for i, _ in enumerate(traj):
                        fix = np.random.uniform(0,1,2)
                        window = self.observe(fix)
                        self.sequence[sub][i] = np.expand_dims(window, 0)

                    seq_bank.addSeq(self.seq[sub].copy(), self.category)

        self.imageCenter = np.array([0.5, 0.5])
        self.seq_bank = seq_bank


    def observe(self, act):
        N, M = self.image.shape
        img = self.image
        fixmap = self.fixmap

        x, y = np.floor(act * np.array([M, N]))

        yStart = max(0, y-RADIUS)
        xStart = max(0, x-RADIUS)

        yEnd = min(y+RADIUS, N)
        xEnd = min(x+RADIUS, M)

        try: #Got a R/G/B image
            window = img[yStart:yEnd, xStart:xEnd, :]

        except IndexError: #Got a B/W image
            window = img[yStart:yEnd, xStart:xEnd]
            #window = img[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS]

        #r_region = fixmap[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS]
        #reward = r_region.mean()

        if window.shape != (2*RADIUS, 2*RADIUS): #ISSUE: Not best way to do this...
            window = np.resize(window, (2*RADIUS, 2*RADIUS))

        return window

    def visualize(self, displayFix= False, subject= None):
        N, M = self.image.shape

        if displayFix is True:
            im = self.fixmap
        else:
            im = self.image

        if subject == None:
            traj = np.concatenate(self.trajs.values())
        else:
            traj = self.trajs[subject]

        center = self.imageCenter

        plt.imshow(-1 * im, cmap= 'Greys', interpolation='nearest')

        plt.scatter(center[0] * M, center[1] * N, c= 'b')
        plt.scatter(traj[:,0] * M, traj[:,1] * N, c= 'r')
        #plt.scatter(traj[:,1], traj[:,0], c= 'b')
        plt.show()

        halt= True


class Episode():
    def __init__(self, rollout, trial = None, subject= None):
        self.rollout = rollout
        self.trial = trial
        self.subject = subject

        self.R = 0.0
        for tran in rollout:
            self.R += tran['reward']


def cropImage(img, fix):
    B = (img == 126).all(axis= 2)
    xs, ys = np.where(np.invert(B) == True)

    crop_img = img[xs[0]:xs[-1], ys[0]:ys[-1]]
    crop_fix = fix[xs[0]:xs[-1], ys[0]:ys[-1]]
    return crop_img, crop_fix

def exactCAT(cat_counts, mode, size= 1.0, asgray= False, get_seqs= None):
    """
    0-49 are training.
    50-74 are val.
    74-99 are test.
    """
    n = np.array(cat_counts.values()).sum()
    if mode == 'train':
        startIdx = 0
    elif mode == 'val':
        startIdx = 50
    elif mode == 'test':
        startIdx = 75

    if get_seqs is not None:
        sb = SequenceBank(categories= cat_counts.keys())
    else:
        sb = None

    sample = []
    c = 0.0
    for cat, count in cat_counts.items():
        stimDir = os.listdir("./CATdata/Stimuli/"+cat)[1:-1]
        trajDir = os.listdir("./CATdata/FIXATIONTRAJS/"+cat)[1:]

        for pick in range(startIdx,startIdx + count):
            c += 1
            if c % 10 == 0:
                print("Image {:,.1f} of {:,.1f}".format(c, n))

            t = loadTrial(pick, cat, stimDir, trajDir, size, asgray, get_seqs, seq_bank = sb)
            sample.append(t)

    if get_seqs is not None:
        return sample, sb

    return sample

def sampleCAT(n = 1000, size= 1.0, asgray= False, categories= None, get_seqs= False):
    """
    """
    #assert type(size) is float
    sample= []
    print "Sample CAT2000 data set..."

    if categories == None:
        categories = os.listdir("./CATdata/Stimuli")[1:]

    if get_seqs is not None:
        sb = SequenceBank()
    else:
        sb = None

    counts= Counter(np.random.choice(categories, n, replace= True))
    c = 0.0
    for cat, count in counts.items():
        stimDir = os.listdir("./CATdata/Stimuli/"+cat)[1:-1]
        trajDir = os.listdir("./CATdata/FIXATIONTRAJS/"+cat)[1:]

        catSize = len(stimDir)
        picks= np.random.choice(np.arange(catSize-1), count, replace= False)

        for pick in picks:
            c += 1
            if c % 10 == 0:
                print("Image {:,.1f} of {:,.1f}".format(c, n))

            t = loadTrial(pick, cat, stimDir, trajDir, size, asgray, get_seqs, seq_bank = sb)
            sample.append(t)
            #IMG = scipy.ndimage.imread('./CATdata/Stimuli/'+cat+'/'+stimDir[pick])
            #fixmap = scipy.ndimage.imread('./CATdata/FIXATIONMAPS/'+cat+'/'+stimDir[pick])
            #IMG = plt.imread('./CATdata/Stimuli/'+cat+'/'+stimDir[pick])
            #fixmap = plt.imread('./CATdata/FIXATIONMAPS/'+cat+'/'+stimDir[pick])

            ##Pre-processing
            #if len(IMG.shape) == 2:
                #IMG = np.asarray(np.dstack((IMG, IMG, IMG)))

            #IMG, fixmap = cropImage(IMG, fixmap)
            #if asgray:
                #IMG = mpl.colors.rgb_to_hsv(IMG)[:,:,2]

            #orig_shape = IMG.shape
            #IMG = scipy.misc.imresize(IMG, size)
            #fixmap = scipy.misc.imresize(fixmap, size)

            #M = scipy.io.loadmat('./CATdata/FIXATIONTRAJS/'+cat+'/'+trajDir[pick])['cellVal'][0][0]
            #trajs = {k[0][0][0][0][0]:k[0][0][0][1] for k in M if k[0][0][0][1].size != 0}

            #t = Trial(cat, trajDir[pick].split('.')[0], IMG, fixmap, trajs, orig_shape, get_seqs= get_seqs)
            #t.removeInvalids() #ISSUE: Assumption

            #t.visualize(displayFix= True)

            #sample.append(t)

    return sample

def loadTrial(pick, cat, stimDir, trajDir, size, asgray, get_seqs, seq_bank = None):
    IMG = plt.imread('./CATdata/Stimuli/'+cat+'/'+stimDir[pick])
    fixmap = plt.imread('./CATdata/FIXATIONMAPS/'+cat+'/'+stimDir[pick])
    salmap = plt.imread('./CATdata/STIMULI/'+cat+'/Output/'+stimDir[pick].split('.')[0]+'_SaliencyMap.jpg')

    #Pre-processing
    if len(IMG.shape) == 2:
        IMG = np.asarray(np.dstack((IMG, IMG, IMG)))

    IMG, fixmap = cropImage(IMG, fixmap)
    if asgray:
        IMG = mpl.colors.rgb_to_hsv(IMG)[:,:,2]

    orig_shape = IMG.shape
    IMG = scipy.misc.imresize(IMG, size)
    fixmap = scipy.misc.imresize(fixmap, size)

    M = scipy.io.loadmat('./CATdata/FIXATIONTRAJS/'+cat+'/'+trajDir[pick])['cellVal'][0][0]
    trajs = {k[0][0][0][0][0]:k[0][0][0][1] for k in M if k[0][0][0][1].size != 0 and k[0][0][0][1].dtype == float}

    t = Trial(cat, trajDir[pick].split('.')[0], IMG, fixmap, salmap, trajs, orig_shape, get_seqs= get_seqs, seq_bank= seq_bank)

    return t

def computeMaxima(data, neighborhood_size, threshold):
    """
    Based on code from:
    http://stackoverflow.com/questions/9111711/get-coordinates-of-local-maxima-in-2d-array-above-certain-value
    """
    sal_N, sal_M = data.shape

    data_max = filters.maximum_filter(data, neighborhood_size)
    maxima = (data == data_max)
    data_min = filters.minimum_filter(data, neighborhood_size)
    diff = ((data_max - data_min) > threshold)
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    xy = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects+1)))
    xy /= np.array([sal_N, sal_M])

    yx = np.fliplr(xy)

    return yx

class SequenceBank(object):
    def __init__(self, categories):
        self.X = None
        self.Y = []
        self.categories = categories

    def addSeq(self, seq, label):
        if self.X is None:
            self.X = np.expand_dims(seq, 0)
        else:
            self.X = np.concatenate((self.X, np.expand_dims(seq, 0)), axis= 0)

        self.Y.append(self.categories.index(label))

    #S = exactCAT(cat_count, mode, size= size, asgray= asgray, get_seqs= get_seqs)

    #X_SEQ = None
    #Y_SEQ = []
    #H, W = S[0].image.shape
    #C = 1

    #N = len(S)
    #X = np.zeros((N, C, H, W))

    #Y = np.zeros(len(S), dtype= int)
    #print "Extracting sequences..."
    #for i, trial in enumerate(S):
        #print "Got sequence: ", i, " of ", len(S)
        #if asgray:
            #X[i] = np.expand_dims(trial.image, 0)
        #else:
            #X[i] = trial.image

        #Y[i] = cat_count.keys().index(trial.category)
        #if get_seqs is not None:
            #for subj, seq in trial.sequence.items():
                #if X_SEQ is None:
                    #X_SEQ = np.expand_dims(seq, 0)
                #else:
                    #X_SEQ = np.concatenate((X_SEQ, np.expand_dims(seq, 0)), axis= 0)

                #Y_SEQ.append(Y[i])

    #print "Extraction complete."
        ##for subject, traj in trial.trajs.items():
            ##for fixation in traj:

    ##X_SEQS = np.array(X_SEQS)
    #Y_SEQ = np.array(Y_SEQ)

    #if shuffle:
        #p = np.random.permutation(range(N))
        #X = X[p]
        #Y = Y[p]

    #if get_seqs is not None:
        #N_SEQ = len(X_SEQ)
        #if shuffle:
            #p = np.random.permutation(range(N_SEQ))
            #X_SEQ = X_SEQ[p]
            #Y_SEQ = Y_SEQ[p]

        #return X_SEQ, Y_SEQ

    #return X, Y