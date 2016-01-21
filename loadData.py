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

def sampleCat(n = 1000):
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
            M = scipy.io.loadmat('./CATdata/FIXATIONTRAJS/'+cat+'/'+trajDir[pick])['cellVal'][0][0]
            trajs = {k[0][0][0][0][0]:k[0][0][0][1] for k in M}

            t = Trial(cat, trajDir[pick].split('.')[0], IMG, trajs)
            sample.append(t)

    return sample

S= sampleCat(n= 100)

halt= True
