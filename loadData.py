import os
import numpy as np
from collections import Counter

import matplotlib as mpl
from matplotlib import pyplot as plt

import math

import sklearn as sk
import scipy
import scipy.io
import scipy.ndimage

RADIUS = 30
HORIZON = 9

##ISSUE: should do more pre-processing. Subtract mean images. etc.

class Trial():
    def __init__(self, cat, idx, img, fixmap, trajs, orig_shape, center_fm= True, get_seqs= False):
        self.category = cat
        self.index = idx

        if img.ndim == 2:
            C = 1 #number of channels
        else:
            C = 3

        self.image = img
        self.fixmap = fixmap
        if center_fm: #ISSUE: Change this when add in color channels?
            fm = fixmap.astype(float)
            self.fixmap = (fm - fm.mean())/fm.std()

        self.trajs = trajs
        self.sequence = {}
        N, M = orig_shape

        #self.imageCenter = np.array([math.floor(img.shape[0]/2), math.floor(img.shape[1]/2)])
        #self.imageCenter = np.array([math.floor(img.shape[1]/2), math.floor(img.shape[0]/2)])
        self.imageCenter = np.array([math.floor(M/2.0), math.floor(N/2.0)])
        for sub, traj in self.trajs.items():
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

            if get_seqs:
                self.sequence[sub] = np.empty((traj.shape[0], C, 2*RADIUS, 2*RADIUS))
                for i, fix in enumerate(traj):
                    #if np.isnan(fix).any():
                        #continue

                    window = self.observe(fix)
                    self.sequence[sub][i] = np.expand_dims(window, 0)

        self.imageCenter = np.array([0.5, 0.5])

    #def removeInvalids(self):
        #r, c = self.image.shape
        ##N, M = self.image.shape

        #for sub, traj in self.trajs.items():
            #tOld = traj.copy()

            #if np.isnan(traj).any():
                #self.trajs.pop(sub)
                #continue

            #traj = np.delete(traj, np.nonzero(traj[:,0] * c < 0), axis= 0)
            #traj = np.delete(traj, np.nonzero(traj[:,1] * r < 0), axis= 0)

            #traj = np.delete(traj, np.nonzero(traj[:,0] * c >= c), axis= 0)
            #traj = np.delete(traj, np.nonzero(traj[:,1] * r >= r), axis= 0)

            #self.trajs[sub] = traj

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
            traj = self.trajs.values()[0]
        else:
            traj = self.trajs[subject]

        center = self.imageCenter

        plt.imshow(im, cmap= 'Greys', interpolation='nearest')
        plt.scatter(center[0] * M, center[1] * N, c= 'b')
        plt.scatter(traj[:,0] * M, traj[:,1] * N, c= 'r')
        #plt.scatter(traj[:,1], traj[:,0], c= 'b')
        plt.show()


class Episode():
    def __init__(self, rollout, trial = None, subject= None):
        self.rollout = rollout
        self.trial = trial
        self.subject = subject

        self.R = 0.0
        for tran in rollout:
            self.R += tran['reward']


def cropImage(img):
    B = (img == 126).all(axis= 2)
    xs, ys = np.where(np.invert(B) == True)
    return img[xs[0]:xs[-1], ys[0]:ys[-1]]

def sampleCAT(n = 1000, size= 1.0, asgray= False, categories= None, get_seqs= False):
    """
    WARNING: resizing the image will mess up the fixation points.

    I should convert human fixations to percentages, because then it will be scale invariant.
    """
    #assert type(size) is float
    sample= []
    print "Sample CAT2000 data set..."

    if categories == None:
        categories = os.listdir("./CATdata/Stimuli")[1:]

    counts= Counter(np.random.choice(categories, n, replace= True))
    for cat, count in counts.items():
        stimDir = os.listdir("./CATdata/Stimuli/"+cat)[1:-1]
        trajDir = os.listdir("./CATdata/FIXATIONTRAJS/"+cat)[1:]

        catSize = len(stimDir)
        picks= np.random.choice(np.arange(catSize-1), count, replace= False)

        for pick in picks:
            #IMG = scipy.ndimage.imread('./CATdata/Stimuli/'+cat+'/'+stimDir[pick])
            #fixmap = scipy.ndimage.imread('./CATdata/FIXATIONMAPS/'+cat+'/'+stimDir[pick])
            IMG = plt.imread('./CATdata/Stimuli/'+cat+'/'+stimDir[pick])
            fixmap = plt.imread('./CATdata/FIXATIONMAPS/'+cat+'/'+stimDir[pick])

            #Pre-processing
            if len(IMG.shape) == 2:
                IMG = np.asarray(np.dstack((IMG, IMG, IMG)))

            IMG = cropImage(IMG)
            if asgray:
                IMG = mpl.colors.rgb_to_hsv(IMG)[:,:,2]

            orig_shape = IMG.shape
            IMG = scipy.misc.imresize(IMG, size)
            fixmap = scipy.misc.imresize(fixmap, size)

            M = scipy.io.loadmat('./CATdata/FIXATIONTRAJS/'+cat+'/'+trajDir[pick])['cellVal'][0][0]
            trajs = {k[0][0][0][0][0]:k[0][0][0][1] for k in M if k[0][0][0][1].size != 0}

            t = Trial(cat, trajDir[pick].split('.')[0], IMG, fixmap, trajs, orig_shape, get_seqs= get_seqs)
            #t.removeInvalids() #ISSUE: Assumption

            sample.append(t)

    return sample
