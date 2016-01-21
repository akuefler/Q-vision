import os
import numpy as np
from collections import Counter


x = 4

def sampleCat(n = 1000):
    categories = os.listdir("./CATdata/Stimuli")[1:]
    counts= Counter(np.random.choice(categories, n, replace= True))
    for cat, count in counts.items():
        catDir = os.listdir("./CATdata/Stimuli/"+cat)
        catSize = len(catDir)
        picks= np.random.choice(np.arange(catSize), count)



sampleCat()
