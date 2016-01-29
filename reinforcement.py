from loadData import *
from matplotlib import pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.decomposition as deco

from networks import *

#from fc_net import TwoLayerNet
#from layers import euclid_log_loss

#from sknn import mlp
#import logging
#logging.basicConfig()

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

        self.N, self.M = self.currImg.shape

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
        self.currPos = act
        self.currPos = np.floor(act * (self.M, self.N)) #ISSUE: Flip N and M?

        assert self.currPos.size != 0

        #if act == 'up':
            #self.currPos[0] += self.step_size

        #elif act == 'down':
            #self.currPos[0] -= self.step_size

        #elif act == 'left':
            #self.currPos[1] -= self.step_size

        #elif act == 'right':
            #self.currPos[1] += self.step_size

        #self.currWin = self.observe(self.currPos[0], self.currPos[1], self.radius)

    def observe(self, trial= None, x= None, y= None, subject= None, display= False):
        if trial == None:
            trial = self.currTrial

        img = trial.image
        fixmap = trial.fixmap

        if x == None:
            x = self.currPos[0]
        if y == None:
            y = self.currPos[1]

        #print('x', x)
        #print('y', y)
        #DISCRETIZE x and y:
        x = np.floor(x)
        y = np.floor(y)

        yStart = max(0, y-RADIUS)
        xStart = max(0, x-RADIUS)

        yEnd = min(y+RADIUS, img.shape[0])
        xEnd = min(x+RADIUS, img.shape[1])

        try: #Got a R/G/B image
            #window = img[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS, :]
            window = img[yStart:yEnd, xStart:xEnd, :]
            cm = None

        except IndexError: #Got a B/W image
            window = img[yStart:yEnd, xStart:xEnd]
            #window = img[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS]
            cm = 'Greys'

        r_region = fixmap[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS]
        if r_region.mean() > RTHRESH:
            reward = REWARD
        else:
            reward = PUNISHMENT

        #plt.imshow(window, cmap= cm, interpolation='nearest')
        #plt.show()

        #ISSUE: Zero-Pad edge cases...
        window_Orig = window

        #if window.shape == (2*RADIUS, 2*RADIUS):
            #if x+RADIUS > img.shape[1]:
                #window= np.column_stack((window, np.zeros((2*RADIUS, 2*RADIUS - window.shape[1]))))

            #if x-RADIUS < 0:
                #window= np.column_stack((np.zeros((2*RADIUS, 2*RADIUS - window.shape[1])), window))

            #if y+RADIUS > img.shape[0]:
            #if y-RADIUS < 0:

        if window.shape != (2*RADIUS, 2*RADIUS): #ISSUE: Not best way to do this...
            window = np.resize(window, (2*RADIUS, 2*RADIUS))

        if display == True:
            plt.imshow(self.currImg, cmap= cm, interpolation='nearest')
            plt.scatter(x, y)
            plt.show()

        return window, reward

    def getEpisodes(self):
        D = []
        for trial in self.sample:
            #trial.visualize()
            for subject, traj in trial.trajs.items():
                #ep = episode(rollout)
                rollout = []
                for i, coord in enumerate(traj[:-1]):
                    if np.isnan(traj[i+1]).any(): ##ISSUE: Maybe a better way to deal with nan
                        continue
                    trans= {}
                    s, _ = self.observe(trial, x= coord[0], y= coord[1])
                    if s.size == 0:
                        continue ##ISSUE: What causes some of these

                    assert s.shape == (2*RADIUS, 2*RADIUS)

                    trans['state'] = s
                    #trans['action'] = coord
                    trans['action'] = np.array([ traj[i+1,0], traj[i+1,1]])
                    #trans['action'] = np.floor(coord) #NOTE: DISCRETIZE.
                    succ, r = self.observe(trial, x= traj[i+1,0], y= traj[i+1,1], subject= subject)
                    trans['reward'] = r
                    trans['successor'] = succ

                    rollout.append(trans)
                ep = Episode(rollout, trial)
                D.append(ep)

        return D

    #def getHumanF(self):
        #F = []
        #for trial in self.sample:
            ##trial.visualize()

            #for subject, traj in trial.trajs.items():
                #for i, coord in enumerate(traj[:-1]):
                    #if np.isnan(traj[i+1]).any(): ##ISSUE: Maybe a better way to deal with nan
                        #continue

                    #trans= {}
                    #s, _ = self.observe(trial, x= coord[0], y= coord[1])
                    #trans['state'] = s
                    #trans['action'] = coord
                    ##trans['action'] = np.floor(coord) #NOTE: DISCRETIZE.
                    #succ, r = self.observe(trial, x= traj[i+1,0], y= traj[i+1,1])

                    #trans['reward'] = r
                    #trans['successor'] = succ

                    #F.append(trans)

        #return F

class FeatureExtractor(object):
    def __init__(self, D):
        self.D = D

    def tranform(self):
        """
        Overwrite method.
        """
        raise NotImplemented

class IdentityExtract(FeatureExtractor):
    def transform(self, X):
        return X


class Agent():
    def __init__(self, environment, phi):
        self.env = environment
        self.phi = phi
        self.D = phi.D

    def act(self, state, epsilon = 0.5):
        legalActs = self.env.getLegalActions()
        act = np.random.choice(legalActs)
        return act

    def extractFeatures(self, window):
        win = window.flatten().reshape(1,-1)
        state = self.phi.transform(win)
        return state

    def cacla(self):
        c_batch_size = 5

        eps = 40

        actor = SigNet(self.D, hidden_size= self.D*2, output_size= 2, batch_size= 50)
        critic = EuclidNet(self.D, hidden_size= self.D*2, output_size= 1, batch_size= c_batch_size)

        while True:

            deltas = np.zeros(c_batch_size)
            X_train = np.empty((c_batch_size, self.D))
            Y_train = np.empty((c_batch_size, 1))

            #Should vectorize this...
            #for i in range(c_batch_size):
                #window, _ = self.env.observe()
                #state = self.extractFeatures(window)
                #a = actor.predict(state)
                #a += np.random.normal(a, eps, 2)

                #self.env.update(a[0])
                #window, r = self.env.observe()
                #succ = self.extractFeatures(window)

                #deltas[i] = r + GAMMA * critic.predict(succ) - critic.predict(state)

            #critic.train(X_train, Y_train, deltas= deltas, batch_size= c_batch_size)

            window, _ = self.env.observe()
            state = self.extractFeatures(window)
            a = actor.predict(state)

            a * (self.env.N, self.env.M)

            a = np.random.normal(a * (self.env.N, self.env.M), eps, 2) / (self.env.M , self.env.N) #ISSUE: Flip?
            a = np.maximum(np.minimum(a, 1),0).reshape(1, 2)

            self.env.update(a[0])
            window, r = self.env.observe(display= True)
            succ = self.extractFeatures(window)

            delta = r + GAMMA * critic.predict(succ) - critic.predict(state)

            #Don't use anything for y. Need to compute gradient of network, not gradient of loss.
            #critic.weight_update(state, np.array([[r]]), delta[0][0]) #ISSUE: Should I use r for y?
            y = r + GAMMA * critic.predict(succ)
            critic.weight_update(state, y)

            print("a: ", a)
            print("Value: ", critic.predict(state))
            print("Delta: ", delta[0][0])

            if delta > 0:
                diff = np.linalg.norm(actor.predict(state) - a)
                actor.weight_update(state, a)

                print("Action: ", actor.predict(state))
                print("Diff: ", diff)


    def simulate(self):
        #for i in range(10):
        while True:
            window = self.env.observe()
            state = self.extractFeatures(window)
            a = self.act(state)
            self.env.update(a)


#mlp = TwoLayerNet(loss_layer=euclid_log_loss)

S = sampleCAT(5, size= 1.0, asgray= True, categories= ['Action', 'Indoor', 'Object', 'Affective'])
#for trial in S:
    #trial.removeInvalid()

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
D = env.getEpisodes()

#pca = getPCA(S2, 10, 100)
#phi = lambda x: pca.transform(x)
#phi = lambda x: x

phi = IdentityExtract((2*RADIUS)**2)

age = Agent(env, phi)
age.cacla()
#age.train(D)
#age.simulate()

#env.observe(x, y, 30)

halt= True