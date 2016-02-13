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

#def getPCA(sample, numWins, k):
    #num_components = (2*RADIUS)**2
    #num_samples = len(sample)*numWins
    #assert num_samples > num_components

    #X = np.zeros((num_samples, num_components))

    ##Create observation matrix:
    #for j, trial in enumerate(sample):
        #img = trial.image
        #try:
            #size_x, size_y, size_z = img.shape
        #except:
            #size_x, size_y = img.shape

        #ys = np.random.randint(RADIUS, size_x-RADIUS, numWins)
        #xs = np.random.randint(RADIUS, size_y-RADIUS, numWins)

        #for i in range(numWins):
            #try:
                #cm = None
                #window = img[ys[i]-RADIUS:ys[i]+RADIUS, xs[i]-RADIUS:xs[i]+RADIUS, :]
            #except:
                #cm = 'Greys'
                #window = img[ys[i]-RADIUS:ys[i]+RADIUS, xs[i]-RADIUS:xs[i]+RADIUS]

            #X[i + numWins*j, :] = window.flatten()

    ##Perform PCA
    #pca = deco.RandomizedPCA(n_components= k)
    #pca.fit(X)

    #return pca


class Environment():
    def __init__(self, sample, pos):
        self.sample = sample
        self.currTrial = sample[0]
        self.currImg = self.currTrial.image
        self.currFixmap = self.currTrial.fixmap

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

    #def update(self, act):
        #self.currPos = act
        #self.currPos = np.floor(act * (self.M, self.N)) #ISSUE: Flip N and M?

        #assert self.currPos.size != 0

        #if act == 'up':
            #self.currPos[0] += self.step_size

        #elif act == 'down':
            #self.currPos[0] -= self.step_size

        #elif act == 'left':
            #self.currPos[1] -= self.step_size

        #elif act == 'right':
            #self.currPos[1] += self.step_size

        #self.currWin = self.observe(self.currPos[0], self.currPos[1], self.radius)

    #def observe(self, trial= None, x= None, y= None, subject= None, display= False):
    def observe(self, act, trial= None, subject= None, display= None):
        if trial == None:
            trial = self.currTrial

        img = trial.image
        fixmap = trial.fixmap

        #if x == None:
            #x = self.currPos[0]
        #if y == None:
            #y = self.currPos[1]

        x, y = np.floor(act * np.array([self.M, self.N]))

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
        reward = r_region.mean()
        #if r_region.mean() > RTHRESH:
            #reward = REWARD
        #else:
            #reward = PUNISHMENT

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

        if display is not None:
            if display == 'img':
                disp = img
            elif display == 'fixmap':
                disp = fixmap

            plt.imshow(disp, cmap= cm, interpolation='nearest')
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

                    act_x = coord[0] / float(self.M)
                    act_y = coord[1] / float(self.N)

                    s, _ = self.observe(act= [act_x, act_y], trial= trial)
                    if s.size == 0:
                        continue ##ISSUE: What causes some of these

                    assert s.shape == (2*RADIUS, 2*RADIUS)
                    trans['state'] = s

                    act_x_next = traj[i+1, 0] / float(self.M)
                    act_y_next = traj[i+1, 1] / float(self.N)

                    #print('act_x_next: ', act_x_next)
                    #print('act_y_next: ', act_y_next)

                    trans['action'] = np.array([act_x_next, act_y_next])
                    succ, r = self.observe(act= [act_x_next, act_y_next], trial= trial, subject= subject)
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

class pcaExtract(FeatureExtractor):
    def train(self, sample, numWins):
        #raise NotImplemented
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
        pca = deco.RandomizedPCA(n_components= self.D)
        pca.fit(X)

        self.pca = pca

    def transform(self, X):
        return self.pca.transform(X)

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

    def getCriticX(self, human_data, shuffle= True):
        X = None
        Y = None
        for ep in human_data:
            for tran in ep.rollout:
                x = self.extractFeatures(tran['state'])
                y = np.array([tran['reward']])
                if X is None:
                    X = x
                else:
                    X = np.row_stack((X, x))

                if Y is None:
                    Y = y
                else:
                    Y = np.concatenate((Y,y))

        if shuffle:
            p = np.random.permutation(range(len(Y)))
            X = X[p]
            Y = Y[p]

        return X, Y



    def cacla(self, human_data, c_batch_size, num_epochs= 10):
        eps = 100

        actor = SigNet(self.D, hidden_size= self.D*2, output_size= 2, batch_size= c_batch_size)
        #critic = EuclidNet(self.D, hidden_size= self.D*2, output_size= 1, batch_size= c_batch_size)
        critic = EuclidNet(self.D, hidden_size= 300, output_size= 1, batch_size= c_batch_size)

        #Pre-train critic on human data.
        #A, b = self.getCriticX(human_data)
        #N = A.shape[0]
        #N_2third = np.floor(2*N/3.0)
        #A_train, b_train = A[:N_2third], b[:N_2third]
        #A_val, b_val = A[N_2third:], b[N_2third:]
        #critic.train(A_train, b_train[np.newaxis].T, A_val, b_val[np.newaxis].T, num_epochs= 5)

        #CACLA
        rewards = []
        for k in range(50):
            deltas = np.zeros((c_batch_size)) #Change to empty for speed-up.
            critic_y = np.zeros((c_batch_size))
            X = np.zeros((c_batch_size, self.D))

            #true_acts = np.zeros((c_batch_size, 2))
            noise_acts = np.zeros((c_batch_size, 2))

            #Should vectorize this...
            for i in range(c_batch_size):
                window, _ = self.env.observe(act= np.random.uniform(0, 1, 2))
                state = self.extractFeatures(window)
                X[i] = state

                #Noise the action in such a way that it remains in the percentage.
                a = actor.predict(state)
                #print("Actor predicted: ", a)
                a = np.random.normal(a * (self.env.M, self.env.N), eps, 2) / (self.env.M , self.env.N) #ISSUE: Flip?
                a = np.maximum(np.minimum(a, 1),0).reshape(1, 2)
                #print("Exploratory action: ", a)

                noise_acts[i] = a

                #self.env.update(a[0])
                window, r = self.env.observe(act= a[0])
                succ = self.extractFeatures(window)

                deltas[i] = r + GAMMA * critic.predict(succ) - critic.predict(state)
                #critic_y[i] = r + GAMMA * critic.predict(succ)
                critic_y[i] = r

                #print('r: ', r)

            #critic.train(X_train, Y_train, deltas= deltas, batch_size= c_batch_size)
            X = sk.preprocessing.scale(X)
            for epoch in range(num_epochs):
                critic.weight_update(X, critic_y.reshape(c_batch_size,1))
                actor.weight_update(X[deltas > 0], noise_acts[deltas > 0])

            window, r = self.env.observe(act= actor.predict(state)[0], display= 'fixmap')
            rewards.append(r)
            print("Iteration: ", k)
            print("GOT REWARD: ", r)

        plt.plot(rewards)
        halt= True


    def simulate(self):
        #for i in range(10):
        while True:
            window = self.env.observe()
            state = self.extractFeatures(window)
            a = self.act(state)
            self.env.update(a)


#mlp = TwoLayerNet(loss_layer=euclid_log_loss)

S = sampleCAT(100, size= 1.0, asgray= True, categories= ['Action', 'Indoor', 'Object', 'Affective'])
#for trial in S:
    #trial.removeInvalid()

S1 = S[:10]
S2 = S[10:]
#S1 = sampleCAT(10, size= 0.3, asgray= True, categories= ['Action', 'Indoor', 'Object', 'Affective'])
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
#phi = pcaExtract(100)
#phi.train(S2, 30)

age = Agent(env, phi)
age.cacla(D, c_batch_size = 100)
#age.train(D)
#age.simulate()

#env.observe(x, y, 30)

halt= True