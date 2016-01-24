from loadData import sampleCAT
from matplotlib import pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.decomposition as deco

from sknn import mlp
import logging
logging.basicConfig()

import math

"""
TODO:
-Figure out how to scale fixation points when I downsample image.
Probably as simple as changing the scale on the x/y axes.
"""

RADIUS = 25
RTHRESH = 120 #Threshold to count as a reward region.

REWARD = 10
PUNISHMENT = -1
GAMMA = 0.95

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
        self.currTrial = sample[0]
        self.currImg = self.currTrial.image

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

    def observe(self, trial= None, x= None, y= None):
        if trial == None:
            trial = self.currTrial

        img = trial.image
        fixmap = trial.fixmap

        if x == None:
            x = self.currPos[0]
        if y == None:
            y = self.currPos[1]

        print('x', x)
        print('y', y)

        try: #Got a R/G/B image
            window = img[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS, :]

            cm = None

        except IndexError: #Got a B/W image
            window = img[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS]

            cm = 'Greys'

        r_region = fixmap[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS]
        if r_region.mean() > RTHRESH:
            reward = REWARD
        else:
            reward = PUNISHMENT

        #plt.imshow(window, cmap= cm, interpolation='nearest')
        #plt.show()
        return window, reward

    def getHumanF(self):
        F = []
        for trial in self.sample:
            #trial.visualize()

            for subject, traj in trial.trajs.items():
                for i, coord in enumerate(traj[:-1]):
                    if np.isnan(traj[i+1]).any(): ##ISSUE: Maybe a better way to deal with nan
                        continue

                    trans= {}
                    s, _ = self.observe(trial, x= coord[0], y= coord[1])
                    trans['state'] = s
                    trans['action'] = np.floor(coord) #NOTE: DISCRETIZE.
                    succ, r = self.observe(trial, x= traj[i+1,0], y= traj[i+1,1])

                    trans['reward'] = r
                    trans['successor'] = succ

                    F.append(trans)

        return F

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
        state = self.phi(win)
        return state

    def NFQ(self, F, numIters = 1000):
        L1 = mlp.Layer('Tanh', units= (2*RADIUS)**2 + 2)
        L2 = mlp.Layer('Tanh', units= math.ceil(1.5*(2*RADIUS)**2 + 2))
        L3 = mlp.Layer('Linear', units= 1)

        classifier = mlp.Regressor([L1, L2, L3])

        for i in range(numIters):
            P = []
            for trans in F:
                #state = self.phi(trans['state'].flatten())
                state = self.extractFeatures(trans['state'])
                action = trans['action']

                #ISSUE: Exhaustive search over all actions...
                max_Q = -float('inf')
                for r in range(10):
                    for c in range(10):
                        act = np.array([r, c]).reshape(1, -1)
                        pred = classifier.predict(np.column_stack((state, act)))
                        if pred > max_Q:
                            max_Q = pred

                target = trans['reward'] + GAMMA*max_Q#max(a, classifier)
                P.append((np.concatenate((state, action)), target))

    def simulate(self):
        #for i in range(10):
        while True:
            window = self.env.observe()
            state = self.extractFeatures(window)
            a = self.act(state)
            self.env.update(a)


S = sampleCAT(5, size= 1.0, asgray= True, categories= ['Action', 'Indoor', 'Object', 'Affective'])
S1 = S[:10]
S2 = S[10:]
S1 = sampleCAT(10, size= 0.3, asgray= True, categories= ['Action', 'Indoor', 'Object', 'Affective'])
try:
    x, y, z = S1[0].image.shape
except:
    x, y = S1[0].image.shape

x = math.floor(y/2.0)
y = math.floor(x/2.0)
env = Environment(S1, [x, y])
F = env.getHumanF()

#pca = getPCA(S2, 10, 100)
#phi = lambda x: pca.transform(x)
phi = lambda x: x

age = Agent(env, phi)
age.NFQ(F)
age.simulate()

#env.observe(x, y, 30)

halt= True