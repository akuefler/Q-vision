from loadData import sampleCAT
from matplotlib import pyplot as plt

import math

class Environment():
    def __init__(self, sample):
        self.sample = sample

        self.currImg = sample[0].image
        #plt.imshow(self.currImg, interpolation='nearest')

    def observe(self, x, y, rad):
        try: #Got a R/G/B image
            window = self.currImg[x-rad:x+rad, y-rad:y+rad, :]
            #window = self.currImg[y-rad:y+rad, x-rad:x+rad, :]
            cm = None
        except IndexError: #Got a B/W image
            window = self.currImg[x-rad:x+rad, y-rad:y+rad]
            #window = self.currImg[y-rad:y+rad, x-rad:x+rad]
            cm = 'Greys'

        plt.imshow(window, cmap= cm, interpolation='nearest')
        plt.show()
        halt = True


class Agent():
    def __init__(self):
        halt= True

S = sampleCAT(10)
env = Environment(S)

try:
    x, y, z = env.currImg.shape
except:
    x, y = env.currImg.shape

x = math.floor(x/2.0)
y = math.floor(y/2.0)

env.observe(x, y, 30)

halt= True