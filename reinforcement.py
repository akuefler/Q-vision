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
from loadData import RADIUS
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
    def __init__(self, sample, use_val):
        #self.sample = sample
        if use_val:
            train_sample, val_sample = np.array_split(sample, [len(sample)*(2.0/3)])
            self.train_sample = train_sample
            self.val_sample = val_sample
        else:
            self.train_sample = sample

        self.currTrial = sample[0]
        #self.currImg = self.currTrial.image
        #self.currFixmap = self.currTrial.fixmap
        self.N, self.M = self.currTrial.image.shape

        #State variables:
        #self.currWin = []

        #Parameters:
        #self.step_size = 60

    def update(self, k, mode= 'train'):
        if mode == 'train':
            self.currTrial = self.train_sample[k]
        elif mode == 'val':
            self.currTrial = self.val_sample[k]
        self.N, self.M = self.currTrial.image.shape

    #def observe(self, trial= None, x= None, y= None, subject= None, display= False):
    def observe(self, act, trial= None, subject= None, display= None):
        if trial == None:
            trial = self.currTrial

        img = trial.image
        fixmap = trial.fixmap

        N, M = img.shape

        #if x == None:
            #x = self.currPos[0]
        #if y == None:
            #y = self.currPos[1]

        x, y = np.floor(act * np.array([M, N]))

        yStart = max(0, y-RADIUS)
        xStart = max(0, x-RADIUS)

        yEnd = min(y+RADIUS, img.shape[0]-1)
        xEnd = min(x+RADIUS, img.shape[1]-1)

        if yEnd - yStart != 2*RADIUS or xEnd - xStart != 2*RADIUS:
            #print ("hit!")
            img = np.pad(img, 2*RADIUS, mode= 'constant', constant_values= PUNISHMENT)
            fixmap = np.pad(fixmap, 2*RADIUS, mode= 'constant', constant_values= PUNISHMENT)

            x += 2*RADIUS
            y += 2*RADIUS

            yStart = max(0, y-RADIUS)
            xStart = max(0, x-RADIUS)

            yEnd = min(y+RADIUS, img.shape[0]-1)
            xEnd = min(x+RADIUS, img.shape[1]-1)

        assert yEnd - yStart == 2*RADIUS
        assert xEnd - xStart == 2*RADIUS

        try: #Got a R/G/B image
            #window = img[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS, :]
            window = img[yStart:yEnd, xStart:xEnd, :]
            cm = None

        except IndexError: #Got a B/W image
            window = img[yStart:yEnd, xStart:xEnd]
            #window = img[y-RADIUS:y+RADIUS, x-RADIUS:x+RADIUS]
            cm = 'Greys'

        r_region = fixmap[yStart:yEnd, xStart:xEnd]
        reward = r_region.mean()

        assert not np.isnan(reward)

        if display is not None:
            if display == 'img':
                disp = img
            elif display == 'fixmap':
                disp = fixmap

            plt.imshow(disp, cmap= cm, interpolation='nearest')
            plt.scatter(x, y)
            plt.scatter([x-RADIUS, x+RADIUS, x-RADIUS, x+RADIUS], [y-RADIUS, y-RADIUS, y+RADIUS, y+RADIUS], color= 'g')
            plt.show()

        return window, reward

    def getEpisodes(self):
        D = []
        for trial in self.train_sample: #ISSUE: What about validation set?
            #trial.visualize()
            for subject, traj in trial.trajs.items():
                #ep = episode(rollout)
                rollout = []
                for i, coord in enumerate(traj[:-1]):
                    if np.isnan(traj[i+1]).any(): ##ISSUE: Maybe a better way to deal with nan
                        continue
                    trans= {}

                    act_x = coord[0]
                    act_y = coord[1]

                    s, _ = self.observe(act= [act_x, act_y], trial= trial)
                    if s.size == 0:
                        continue ##ISSUE: What causes some of these

                    assert s.shape == (2*RADIUS, 2*RADIUS)
                    trans['state'] = s

                    act_x_next = traj[i+1, 0]
                    act_y_next = traj[i+1, 1]

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
        print "Training PCA extractor..."
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

    def attendToImage(self, imageIdx, c_batch_size, num_epochs= 10,  mode= 'train'):
        self.env.update(imageIdx, mode= mode)
        rewards = []
        eps = 100

        start_coord = np.random.uniform(0, 1, 2)
        for t in range(HORIZON):
            if t == 0:
                window, _ = self.env.observe(act= start_coord) #Just picking points randomly...

            if mode == 'train':
                deltas = np.zeros((c_batch_size)) #Change to empty for speed-up.
                critic_y = np.zeros((c_batch_size))
                X = np.zeros((c_batch_size, self.D))
                noise_acts = np.zeros((c_batch_size, 2))

                #Should vectorize this...
                #if t == 0:
                    #window, _ = self.env.observe(act= start_coord) #Just picking points randomly...

                ##Create batch by taking different exploratory actions.
                for i in range(c_batch_size):
                    state = self.extractFeatures(window)
                    X[i] = state

                    #Noise the action in such a way that it remains in the percentage.
                    a = self.actor.predict(state)
                    a = np.random.normal(a * (self.env.M, self.env.N), eps, 2) / (self.env.M , self.env.N)
                    a = np.maximum(np.minimum(a, 1),0).reshape(1, 2)

                    noise_acts[i] = a
                    window, r = self.env.observe(act= a[0])
                    succ = self.extractFeatures(window)

                    deltas[i] = r + GAMMA * self.critic.predict(succ) - self.critic.predict(state)
                    critic_y[i] = r + GAMMA * self.critic.predict(succ)

                X = sk.preprocessing.scale(X)
                for epoch in range(num_epochs):
                    self.critic.weight_update(X, critic_y.reshape(c_batch_size,1))
                    self.actor.weight_update(X[deltas > 0], noise_acts[deltas > 0])
            elif mode == 'val':
                state = self.extractFeatures(window)

            window, r = self.env.observe(act= self.actor.predict(state)[0])
            rewards.append(r)

        return rewards

        #start_coord = np.random.uniform(0, 1, 2)
        #for t in range(HORIZON):

            ##Pick random starting point on image.
            #if t == 0:
                #window, _ = self.env.observe(act= start_coord)
                #state = self.extractFeatures(window)

            #if mode == 'train':
                #deltas = np.zeros((c_batch_size)) #Change to empty for speed-up.
                #critic_y = np.zeros((c_batch_size))
                #X = np.zeros((c_batch_size, self.D))
                #noise_acts = np.zeros((c_batch_size, 2))

                ##Should vectorize this...

                ###Create batch by taking different exploratory actions.
                ###Maybe it doesn't really make sense to do batches...
                #win = window
                #for i in range(c_batch_size):
                    ##prev_state = self.extractFeatures(window)
                    ##X[i] = prev_state #ISSUE: This is the same example every time, make note.
                    #X[i] = self.extractFeatures(win)
                    #state = np.expand_dims(X[i], 0)

                    ##Noise the action in such a way that it remains in the percentage.
                    #a = self.actor.predict(state)
                    #a = np.random.normal(a * (self.env.M, self.env.N), eps, 2) / (self.env.M , self.env.N)
                    #a = np.maximum(np.minimum(a, 1),0).reshape(1, 2)

                    #noise_acts[i] = a
                    #win, r = self.env.observe(act= a[0])
                    #succ = self.extractFeatures(win)

                    #deltas[i] = r + GAMMA * self.critic.predict(succ) - self.critic.predict(state)
                    #critic_y[i] = r + GAMMA * self.critic.predict(succ)

                #X = sk.preprocessing.scale(X) ##If X repeated examples, don't do this!
                #for epoch in range(num_epochs):
                    #self.critic.weight_update(X, critic_y.reshape(c_batch_size,1))
                    #self.actor.weight_update(X[deltas > 0], noise_acts[deltas > 0])

            #elif mode == 'val':
                #pass

            #else:
                #assert False

            ##Update window with learned best action.
            #window, r = self.env.observe(act= self.actor.predict(state)[0])
            #rewards.append(r)

            #return rewards

    def cacla(self, human_data, c_batch_size, num_epochs= 10):
        print "Performing CACLA..."
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
        print "Starting iterations..."
        #rewards = []

        num_iters = 100

        mean_train_rewards = np.zeros(num_iters)
        mean_val_rewards = np.zeros(num_iters)

        for it in range(num_iters):
            train_rewards = []
            val_rewards = []
            print("ITERATION: ", it)

            trainpoints = np.random.choice(range(len(self.env.train_sample)), 3)

            for d in trainpoints:
                self.env.update(d, mode= 'train')
                print("Image: ", d)

                start_coord = np.random.uniform(0, 1, 2)
                for t in range(HORIZON):

                    deltas = np.zeros((c_batch_size)) #Change to empty for speed-up.
                    critic_y = np.zeros((c_batch_size))
                    X = np.zeros((c_batch_size, self.D))
                    noise_acts = np.zeros((c_batch_size, 2))

                    #Should vectorize this...
                    if t == 0:
                        window, _ = self.env.observe(act= start_coord) #Just picking points randomly...

                    ##Create batch by taking different exploratory actions.
                    for i in range(c_batch_size):
                        state = self.extractFeatures(window)
                        X[i] = state

                        #Noise the action in such a way that it remains in the percentage.
                        a = actor.predict(state)
                        #print("Actor predicted: ", a)
                        a = np.random.normal(a * (self.env.M, self.env.N), eps, 2) / (self.env.M , self.env.N) #ISSUE: Flip?
                        a = np.maximum(np.minimum(a, 1),0).reshape(1, 2)
                        #print("Exploratory action: ", a)

                        noise_acts[i] = a
                        window, r = self.env.observe(act= a[0])
                        succ = self.extractFeatures(window)

                        deltas[i] = r + GAMMA * critic.predict(succ) - critic.predict(state)
                        critic_y[i] = r + GAMMA * critic.predict(succ)

                    X = sk.preprocessing.scale(X)
                    for epoch in range(num_epochs):
                        critic.weight_update(X, critic_y.reshape(c_batch_size,1))
                        actor.weight_update(X[deltas > 0], noise_acts[deltas > 0])

                    window, r = self.env.observe(act= actor.predict(state)[0])
                    train_rewards.append(r)
                    #print("Time-Step: ", t)
                    #print("GOT REWARD: ", r)

            #plt.plot(rewards)
            valpoints = np.random.choice(range(len(self.env.val_sample)), 3)

            for d in valpoints:
                self.env.update(d, mode= 'val')
                start_coord = np.random.uniform(0, 1, 2)
                for t in range(HORIZON):
                    if t == 0:
                        window, _ = self.env.observe(act= start_coord)
                    state = self.extractFeatures(window)
                    a = actor.predict(state)
                    window, r = self.env.observe(act= a[0])
                    val_rewards.append(r)

            mean_train_rewards[it] = np.array(train_rewards).mean()
            mean_val_rewards[it] = np.array(val_rewards).mean()
            halt= True

        plt.plot(mean_train_rewards, c= 'b')
        plt.plot(mean_val_rewards, c= 'r')
        plt.show()

        halt = True



    def simulate(self):
        #for i in range(10):
        while True:
            window = self.env.observe()
            state = self.extractFeatures(window)
            a = self.act(state)
            self.env.update(a)


if False:
    S = sampleCAT(200, size= 0.5, asgray= True, categories= ['Action', 'Indoor', 'Sketch', 'Object', 'Affective'])
    S1 = S[:10]
    S2 = S[10:]
    env = Environment(S1)
    #D = env.getEpisodes()
    phi = pcaExtract(100)
    phi.train(S2, 50)
    age = Agent(env, phi)
    age.cacla(None, c_batch_size = 100)
else:
    S = sampleCAT(15, size= (256, 256), asgray= True, categories= ['Action', 'Indoor', 'Sketch', 'Object', 'Affective'])
    env = Environment(S, use_val = True)
    #D = env.getEpisodes()
    phi = IdentityExtract((2*RADIUS)**2)
    age = Agent(env, phi)
    age.cacla(None, c_batch_size = 100)


#S = sampleCAT(150, size= (256, 256), asgray= True, categories= ['Action', 'Indoor', 'Sketch', 'Object', 'Affective'])

halt= True