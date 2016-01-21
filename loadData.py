import os
import numpy as np
from collections import Counter

import scipy
import scipy.io
import scipy.ndimage


x = 4

class Trial():
    def __init__(self, cat, idx, img, trajs):
        self.category = cat
        self.index = idx
        self.image = img
        self.trajs = trajs

def cropImage(img):
    B = (img == 126).all(axis= 2)
    xs, ys = np.where(np.invert(B) == True)
    return img[xs[0]:xs[-1], ys[0]:ys[-1]]

def sampleCAT(n = 1000):
    sample= []

    categories = os.listdir("./CATdata/Stimuli")[1:]
    counts= Counter(np.random.choice(categories, n, replace= True))
    for cat, count in counts.items():
        stimDir = os.listdir("./CATdata/Stimuli/"+cat)[1:-1]
        trajDir = os.listdir("./CATdata/FIXATIONTRAJS/"+cat)[1:]

        catSize = len(stimDir)
        picks= np.random.choice(np.arange(catSize), count, replace= False)

        for pick in picks:
            IMG = scipy.ndimage.imread('./CATdata/Stimuli/'+cat+'/'+stimDir[pick])

            #Mild pre-processing
            if len(IMG.shape) == 2:
                IMG = np.expand_dims(IMG, 2)

            IMG = cropImage(IMG)
            M = scipy.io.loadmat('./CATdata/FIXATIONTRAJS/'+cat+'/'+trajDir[pick])['cellVal'][0][0]
            trajs = {k[0][0][0][0][0]:k[0][0][0][1] for k in M}

            t = Trial(cat, trajDir[pick].split('.')[0], IMG, trajs)
            sample.append(t)

    return sample
