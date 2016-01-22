from loadData import sampleCAT
from matplotlib import pyplot as plt
import numpy as np
import sklearn.decomposition as deco

import math

RADIUS = 15

def getPCA(sample, numWins, k):
    num_components = (2*RADIUS)**2
    num_samples = len(sample)*numWins
    assert num_samples > num_components

    X = np.zeros((num_samples, num_components))

    #Create observation matrix:
    for j, trial in enumerate(sample):
        img = trial.image
        try:
            size_x, size_y, size_z = img.shape
        except:
            size_x, size_y = img.shape

        ys = np.random.randint(RADIUS, size_x-RADIUS, numWins)
        xs = np.random.randint(RADIUS, size_y-RADIUS, numWins)

        for i in range(numWins):
            try:
                cm = None
                window = img[ys[i]-RADIUS:ys[i]+RADIUS, xs[i]-RADIUS:xs[i]+RADIUS, :]
            except:
                cm = 'Greys'
                window = img[ys[i]-RADIUS:ys[i]+RADIUS, xs[i]-RADIUS:xs[i]+RADIUS]

            X[i + numWins*j, :] = window.flatten()

    #Perform PCA
    pca = deco.RandomizedPCA(n_components= k)
    pca.fit(X)

    return pca

class Environment():
    def __init__(self, sample, pos):
        self.sample = sample
        self.currImg = sample[0].image

        #State variables:
        #self.currWin = []
        self.currPos = pos

        #Parameters:
        self.step_size = 60
        #plt.imshow(self.currImg, interpolation='nearest')

    def getLegalActions(self):
        legalActions = ['up', 'down', 'left', 'right']
        if self.currPos[0] + (RADIUS+self.step_size) > self.currImg.shape[0]:
            legalActions.remove('up')

        elif self.currPos[0] - (RADIUS+self.step_size) < 0:
            legalActions.remove('down')

        elif self.currPos[1] + (RADIUS+self.step_size) > self.currImg.shape[1]:
            legalActions.remove('right')

        elif self.currPos[1] - (RADIUS+self.step_size) < 0:
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

    def observe(self, x= None, y= None):
        if x == None:
            x = self.currPos[0]
        if y == None:
            y = self.currPos[1]

        try: #Got a R/G/B image
            #window = self.currImg[x-rad:x+rad, y-rad:y+rad, :]
            window = self.currImg[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS, :]
            cm = None
        except IndexError: #Got a B/W image
            #window = self.currImg[x-rad:x+rad, y-rad:y+rad]
            window = self.currImg[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS]
            cm = 'Greys'

        plt.imshow(window, cmap= cm, interpolation='nearest')
        plt.show()
        return window


class Agent():
    def __init__(self, environment, phi):
        self.env = environment
        self.phi = phi

    def act(self, state, epsilon = 0.5):
        legalActs = self.env.getLegalActions()
        act = np.random.choice(legalActs)
        return act

    def extractFeatures(self, window):
        win = window.flatten().reshape(1,-1)
        state = self.phi.transform(win)
        return state

    def simulate(self):
        #for i in range(10):
        while True:
            window = self.env.observe()
            state = self.extractFeatures(window)
            a = self.act(state)
            self.env.update(a)


S = sampleCAT(130, size= 0.3, asgray= True, categories= ['Action', 'Indoor', 'Object', 'Affective'])
S1 = S[:10]
S2 = S[10:]
try:
    x, y, z = S[0].image.shape
except:
    x, y = S[0].image.shape

x = math.floor(y/2.0)
y = math.floor(x/2.0)

pca = getPCA(S2, 10, 100)
env = Environment(S1, [x, y])

age = Agent(env, pca)
age.simulate()

#env.observe(x, y, 30)

halt= True