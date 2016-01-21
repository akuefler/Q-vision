from loadData import sampleCAT
from matplotlib import pyplot as plt
import numpy as np

import math

class Environment():
    def __init__(self, sample, pos):
        self.sample = sample
        self.currImg = sample[0].image

        #State variables:
        #self.currWin = []
        self.currPos = pos

        #Parameters:
        self.step_size = 60
        self.radius = 80
        #plt.imshow(self.currImg, interpolation='nearest')

    def getLegalActions(self):
        legalActions = ['up', 'down', 'left', 'right']
        if self.currPos[0] + (self.radius+self.step_size) > self.currImg.shape[0]:
            legalActions.remove('up')

        elif self.currPos[0] - (self.radius+self.step_size) < 0:
            legalActions.remove('down')

        elif self.currPos[1] + (self.radius+self.step_size) > self.currImg.shape[1]:
            legalActions.remove('right')

        elif self.currPos[1] - (self.radius+self.step_size) < 0:
            legalActions.remove('left')

        return legalActions

    def update(self, act):
        if act == 'up':
            self.currPos[0] += self.step_size

        elif act == 'down':
            self.currPos[0] -= self.step_size

        elif act == 'left':
            self.currPos[1] -= self.step_size

        elif act == 'right':
            self.currPos[1] += self.step_size

        #self.currWin = self.observe(self.currPos[0], self.currPos[1], self.radius)


    def observe(self, x= None, y= None, rad= None):
        if x == None:
            x = self.currPos[0]
        if y == None:
            y = self.currPos[1]
        if rad == None:
            rad = self.radius

        try: #Got a R/G/B image
            #window = self.currImg[x-rad:x+rad, y-rad:y+rad, :]
            window = self.currImg[y-rad:y+rad, x-rad:x+rad, :]
            cm = None
        except IndexError: #Got a B/W image
            #window = self.currImg[x-rad:x+rad, y-rad:y+rad]
            window = self.currImg[y-rad:y+rad, x-rad:x+rad]
            cm = 'Greys'

        plt.imshow(window, cmap= cm, interpolation='nearest')
        plt.show()
        return window


class Agent():
    def __init__(self, environment):
        self.env = environment

    def simulate(self):
        #for i in range(10):
        while True:
            window = self.env.observe()
            a = self.act(window)
            self.env.update(a)

    def act(self, state, epsilon = 0.5):
        legalActs = self.env.getLegalActions()
        act = np.random.choice(legalActs)
        return act


S = sampleCAT(10, categories= ['Action', 'Indoor', 'Object', 'Affective'])
try:
    x, y, z = S[0].image.shape
except:
    x, y = S[0].image.shape

x = math.floor(y/2.0)
y = math.floor(x/2.0)

env = Environment(S, [x, y])

age = Agent(env)
age.simulate()

#env.observe(x, y, 30)

halt= True