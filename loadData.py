import os
import numpy as np
from collections import Counter

import matplotlib as mpl
from matplotlib import pyplot as plt

import math

import scipy
import scipy.io
import scipy.ndimage

##ISSUE: should do more pre-processing. Subtract mean images. etc.

class Trial():
    def __init__(self, cat, idx, img, fixmap, trajs):
        self.category = cat
        self.index = idx

        self.image = img
        self.fixmap = fixmap

        self.trajs = trajs

        #self.imageCenter = np.array([math.floor(img.shape[0]/2), math.floor(img.shape[1]/2)])
        self.imageCenter = np.array([math.floor(img.shape[1]/2), math.floor(img.shape[0]/2)])
        for key, val in self.trajs.items():
            self.trajs[key] -= np.mean(self.trajs[key], axis= 0)
            self.trajs[key] += self.imageCenter

    def visualize(self, displayFix= False):
        if displayFix is True:
            im = self.fixmap
        else:
            im = self.image

        traj = self.trajs.values()[0]
        center = self.imageCenter

        plt.imshow(im, cmap= 'Greys', interpolation='nearest')
        plt.scatter(center[0], center[1], c= 'b')
        plt.scatter(traj[:,0], traj[:,1], c= 'r')
        #plt.scatter(traj[:,1], traj[:,0], c= 'b')
        plt.show()


def cropImage(img):
    B = (img == 126).all(axis= 2)
    xs, ys = np.where(np.invert(B) == True)
    return img[xs[0]:xs[-1], ys[0]:ys[-1]]

def sampleCAT(n = 1000, size= 1.0, asgray= False, categories= None):
    """
    WARNING: resizing the image will mess up the fixation points.
    """
    assert type(size) is float
    sample= []

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

            #IMG = scipy.misc.imresize(IMG, size)
            #fixmap = scipy.misc.imresize(fixmap, size)

            M = scipy.io.loadmat('./CATdata/FIXATIONTRAJS/'+cat+'/'+trajDir[pick])['cellVal'][0][0]
            trajs = {k[0][0][0][0][0]:k[0][0][0][1] for k in M}

            t = Trial(cat, trajDir[pick].split('.')[0], IMG, fixmap, trajs)
            sample.append(t)

    return sample
