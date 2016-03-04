from loadData import *
from matplotlib import pyplot as plt
import numpy as np
import sklearn as sk
import sklearn.decomposition as deco

from matplotlib import patches

import time
import copy

from networks import *

#from fc_net import TwoLayerNet
#from layers import euclid_log_loss

#from sknn import mlp
#import logging
#logging.basicConfig()

import math

from loadData import RADIUS
RTHRESH = 120 #Threshold to count as a reward region.

REWARD = 10
PUNISHMENT = -1
GAMMA = 0.95

RADIUS = 10

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

def loadRewardPlots(path, scaling_factor):
    mtr = np.load(path+'/mean_train_rewards.npz')
    mean_train_rewards = mtr.items()[0][1]

    mrr = np.load(path+'/mean_rand_rewards.npz')
    mean_rand_rewards = mrr.items()[0][1]

    mvr = np.load(path+'/mean_val_rewards.npz')
    mean_val_rewards = mvr.items()[0][1]

    ctr = np.load(path+'/cum_train_rewards.npz')
    cum_train_rewards = ctr.items()[0][1]

    cvr = np.load(path+'/cum_val_rewards.npz')
    cum_val_rewards = cvr.items()[0][1] * scaling_factor

    crr = np.load(path+'/cum_rand_rewards.npz')
    cum_rand_rewards = crr.items()[0][1]

    #tctr = np.load(path+'/total_cum_train_rewards.npz')
    #total_cum_train_rewards = tctr.items()[0][1]

    #tcvr = np.load(path+'/total_cum_val_rewards.npz')
    #total_cum_val_rewards = tcvr.items()[0][1]

    #tcrr = np.load(path+'/total_cum_rand_rewards.npz')
    #total_cum_rand_rewards = tcrr.items()[0][1]

    total_cum_train_rewards = np.zeros_like(cum_train_rewards)
    total_cum_val_rewards = np.zeros_like(cum_val_rewards)
    total_cum_rand_rewards = np.zeros_like(cum_rand_rewards)
    for it in range(len(cum_train_rewards)):
        total_cum_train_rewards[it] = cum_train_rewards[:it+1].mean()
        total_cum_val_rewards[it] = cum_val_rewards[:it+1].mean()
        total_cum_rand_rewards[it] = cum_rand_rewards[:it+1].mean()

    f, (ax1, ax3) = plt.subplots(2, 1)
    ax1.plot(mean_train_rewards, c= 'b')
    ax1.plot(mean_val_rewards, c= 'r')
    ax1.plot(mean_rand_rewards, c= 'g')

    #ax2.plot(cum_train_rewards, c= 'b')
    #ax2.plot(cum_val_rewards, c= 'r')
    #ax2.plot(cum_rand_rewards, c= 'g')

    ax3.plot(total_cum_train_rewards, c= 'b')
    ax3.plot(total_cum_val_rewards, c= 'r')
    ax3.plot(total_cum_rand_rewards, c= 'g')

    plt.show()

class Environment():
    def __init__(self, sample, sample_val= None):
        #self.sample = sample
        if sample_val is None:
            train_sample, val_sample = np.array_split(sample, [len(sample)*(2.0/3)])
            self.train_sample = train_sample
            self.val_sample = val_sample
        else:
            self.train_sample = sample
            self.val_sample = sample_val

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
    def observe(self, act, fixmap, trial= None, subject= None, display= None, delete_reward= False):
        if trial == None:
            trial = self.currTrial

        img = trial.image

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

        r_region = fixmap[yStart:yEnd, xStart:xEnd].copy()
        if delete_reward:
            fixmap[yStart:yEnd, xStart:xEnd] = fixmap.min()
        reward = r_region.mean()

        assert not np.isnan(reward)

        if display is not None:
            if display == 'img':
                disp = -1 * img
            elif display == 'fixmap':
                disp = -1 * fixmap

            f = plt.figure()
            ax = f.add_subplot(111)
            ax.imshow(disp, cmap= cm, interpolation= 'nearest')
            ax.add_patch(
                patches.Rectangle([x-RADIUS, y-RADIUS],2*RADIUS,2*RADIUS, color= 'r', linewidth= 2, fill= False)
            )

            plt.show()
            #plt.imshow(disp, cmap= cm, interpolation='nearest')
            #plt.scatter(x, y)
            #plt.scatter([x-RADIUS, x+RADIUS, x-RADIUS, x+RADIUS], [y-RADIUS, y-RADIUS, y+RADIUS, y+RADIUS], color= 'g')
            #plt.show()

        return window, reward, fixmap

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

    def transform(self):
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
        #assert num_samples > num_components

        X = np.zeros((num_samples, num_components))
        D = self.D[0]

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
        pca = deco.PCA(n_components= D)
        pca.fit(X)

        self.pca = pca

    def transform(self, X):
        return self.pca.transform(X.flatten().reshape(1,-1))

class FlattenExtract(FeatureExtractor):
    def transform(self, X):
        X = X.flatten().reshape(1,-1)
        return X

class IdentityExtract(FeatureExtractor):
    def transform(self, X):
        if X.ndim == 2:
            X = X[np.newaxis]
        return X[np.newaxis]


class Agent():
    def __init__(self, environment, actor, critic= None, phi= FlattenExtract):
        self.actor = actor
        self.critic = critic
        self.env = environment
        self.phi = phi

        self.D = phi.D

    def act(self, state, epsilon = 0.5):
        legalActs = self.env.getLegalActions()
        act = np.random.choice(legalActs)
        return act

    def batch_cacla(self, ix, c_batch_size, num_epochs, mode = 'train'):
        self.env.update(ix, mode= mode)
        #eps = 100
        #eps = 15

        #start_coord = np.random.uniform(0, 1, 2)
        start_coord = np.array([0.5, 0.5])
        arr = np.array([float(i)/MEMORY for i in range(MEMORY + 1)])

        #rewards = []
        #for t in range(HORIZON):
            #if t == 0:
                #window, _ = self.env.observe(act= start_coord)
            #state = self.extractFeatures(window)
            #window, r = self.env.observe(act= self.actor.predict(state)[0])
            #rewards.append(r)

        #halt= True

        if mode == 'train':

            deltas = np.zeros((c_batch_size * HORIZON)) #Change to empty for speed-up.
            critic_y = np.zeros((c_batch_size * HORIZON))
            X = np.zeros((c_batch_size * HORIZON,) + self.D)
            M = np.zeros((c_batch_size * HORIZON, MEMORY**2))
            noise_acts = np.zeros((c_batch_size * HORIZON, 2))

            for b in range(c_batch_size):

                a_prev = None

                m = np.zeros((MEMORY, MEMORY))
                reward_region = copy.copy(self.env.currTrial.fixmap)
                for t in range(HORIZON):
                    i = b*HORIZON + t

                    if t == 0:
                        window, _, reward_region = self.env.observe(act= start_coord, fixmap= reward_region)
                    state = self.phi.transform(window)
                    mem = m.reshape((1, MEMORY**2)).copy()
                    #state = np.concatenate((state, m.reshape((1, MEMORY**2))), axis= 1)

                    X[i] = state
                    M[i] = mem

                    #Noise the action in such a way that it remains in the percentage.
                    a = self.actor.predict(state, mem)
                    a = np.random.normal(a * (self.env.M, self.env.N), EPSILON, 2) / (self.env.M , self.env.N) #ISSUE: Flip?
                    a = np.maximum(np.minimum(a, 1),0).reshape(1, 2)

                    noise_acts[i] = a
                    window, r, reward_region = self.env.observe(act= a[0], fixmap= reward_region, delete_reward= True)
                    succ = self.phi.transform(window)

                    #if a_prev is not None:
                        #if np.linalg.norm(a - a_prev) < 0.25:
                            #r -= 200

                    #Update m here:
                    H, _, _ = np.histogram2d([a[0, 0]], [a[0, 1]], bins=(arr, arr))
                    m += H.astype('float32')
                    mem_succ = m.reshape((1, MEMORY**2))

                    #succ = np.concatenate((succ, m.reshape((1, MEMORY**2))), axis= 1)

                    deltas[i] = r + GAMMA * self.critic.predict(succ, mem_succ) - self.critic.predict(state, mem)
                    critic_y[i] = r + GAMMA * self.critic.predict(succ, mem_succ)

                    #a_prev = a

            m = np.zeros((MEMORY, MEMORY))
            for epoch in range(num_epochs):
                self.critic.weight_update(X, M, critic_y.reshape(HORIZON*c_batch_size,1))
                self.actor.weight_update(X[deltas > 0], M[deltas > 0], noise_acts[deltas > 0])

            #print "Num positive states: ", X[deltas > 0].shape[0]

        rewards, _ = self.simulate(start_coord=start_coord)
        return rewards
        #rewards = []
        #m = np.zeros((MEMORY, MEMORY))
        #for t in range(HORIZON):
            #if t == 0:
                #window, _ = self.env.observe(act= start_coord)
            #state = self.phi.transform(window)

            ##state = np.concatenate((state, m.reshape((1, MEMORY**2))), axis= 1)
            #mem = m.reshape((1, MEMORY**2))

            #a = self.actor.predict(state, mem)[0]
            #H, _, _ = np.histogram2d([a[0]], [a[1]], bins=(arr, arr))
            #m += H.astype('float32')

            #window, r = self.env.observe(act= a)
            #rewards.append(r)

        #return rewards


    def simulate(self, agent= None, mode= 'train', start_coord= None, imgIdx= None, display= None, verbose= False):
        rewards = []
        actions = []
        arr = np.array([float(i)/MEMORY for i in range(MEMORY + 1)])

        if agent is None:
            agent = self.actor

        if imgIdx is not None:
            self.env.update(imgIdx, mode= 'train')

        if start_coord is None:
            #start_coord = np.random.uniform(0, 1, 2)
            start_coord = np.array([0.5, 0.5])

        m = np.zeros((MEMORY, MEMORY))
        reward_region = copy.copy(self.env.currTrial.fixmap)
        for t in range(HORIZON):
            if t == 0:
                window, _, reward_region = self.env.observe(act= start_coord, fixmap= reward_region)
            state = self.phi.transform(window)

            #state = np.concatenate((state, m.reshape((1, MEMORY**2))), axis= 1)
            mem = m.reshape((1, MEMORY**2))

            a = agent.predict(state, mem)[0]
            if verbose:
                print 'Action was: ', a
            actions.append(a)

            H, _, _ = np.histogram2d([a[0]], [a[1]], bins=(arr, arr))
            m += H.astype('float32')

            window, r, reward_region = self.env.observe(act= a, fixmap= reward_region, display= display, delete_reward= True)
            rewards.append(r)

        return rewards, actions


    #def cacla(self, ix, mode, c_batch_size, num_epochs):
        #self.env.update(ix, mode= mode)
        #eps = 100
        ##print("Image: ", d)

        #start_coord = np.random.uniform(0, 1, 2)

        #rewards = []

        #for t in range(HORIZON):
            #if t == 0:
                #window, _ = self.env.observe(act= start_coord) #Just picking points randomly...

                ##tru_state = self.extractFeatures(window)
                ##win = window
            #if mode == 'train':
                #tru_state = self.extractFeatures(window)
                #win = window

                #deltas = np.zeros((c_batch_size)) #Change to empty for speed-up.
                #critic_y = np.zeros((c_batch_size))
                #X = np.zeros((c_batch_size, self.D))
                #noise_acts = np.zeros((c_batch_size, 2))

                ###Create batch by taking different exploratory actions.
                #for i in range(c_batch_size):
                    #state = self.extractFeatures(win)
                    #X[i] = state

                    ##Noise the action in such a way that it remains in the percentage.
                    #a = self.actor.predict(state)
                    ##print("Actor predicted: ", a)
                    #a = np.random.normal(a * (self.env.M, self.env.N), eps, 2) / (self.env.M , self.env.N) #ISSUE: Flip?
                    #a = np.maximum(np.minimum(a, 1),0).reshape(1, 2)
                    ##print("Exploratory action: ", a)

                    #noise_acts[i] = a
                    #win, r = self.env.observe(act= a[0])
                    #succ = self.extractFeatures(win)

                    #deltas[i] = r + GAMMA * self.critic.predict(succ) - self.critic.predict(state)
                    #critic_y[i] = r + GAMMA * self.critic.predict(succ)

                ##X = sk.preprocessing.scale(X) ##ISSUE: Shouldn't do this, because can't replicate it at test time.
                #for epoch in range(num_epochs):
                    #self.critic.weight_update(X, critic_y.reshape(c_batch_size,1))
                    #self.actor.weight_update(X[deltas > 0], noise_acts[deltas > 0])

                #window, r = self.env.observe(act= self.actor.predict(tru_state)[0])
                #rewards.append(r)

            #elif mode == 'val':
                #state = self.extractFeatures(window)
                #a = self.actor.predict(state)
                #window, r = self.env.observe(act= a[0])
                #rewards.append(r)

        #return rewards

    def train(self, human_data, c_batch_size, num_iters= 200, num_epochs= 1, start_iter= 0, display= False, save= False):
        print "Performing CACLA..."
        reg = 1e-2

        D = self.D[0]
        train_size = len(self.env.train_sample)
        val_size = len(self.env.val_sample)
        hid_size = int(np.prod(self.D)*(4.0/3))

        #if arc == 0:
            #self.actor = SigNet(D, MEMORY**2, hidden_size= hid_size, output_size= 2, batch_size= c_batch_size*HORIZON)
            #self.critic = EuclidNet(D, MEMORY**2, hidden_size= hid_size, output_size= 1, batch_size= c_batch_size*HORIZON)

            #random_actor = SigNet(D, MEMORY**2, hidden_size= hid_size, output_size= 2, batch_size= c_batch_size*HORIZON)

        #elif arc == 1:
            #self.actor = SplitNet(self.D, MEMORY**2, hidden_size= hid_size, output= 'sig', output_size= 2, batch_size= c_batch_size*HORIZON)
            #self.critic = SplitNet(D, MEMORY**2, hidden_size= hid_size, output= 'euclid', output_size= 1, batch_size= c_batch_size*HORIZON)

            #random_actor = SplitNet(self.D, MEMORY**2, hidden_size= hid_size, output_size= 2, batch_size= c_batch_size*HORIZON)

        #elif arc == 2:
            #self.actor = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, hidden_size= hid_size, output_size= 2, output= 'sig', batch_size= c_batch_size*HORIZON, reg= reg)
            #self.critic = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, hidden_size= hid_size, output_size= 1, output= 'euclid', batch_size= c_batch_size*HORIZON, reg= reg)

            #random_actor = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, hidden_size= hid_size, output_size= 2, output= 'sig', batch_size= c_batch_size*HORIZON, reg= reg)
        #else:
            #assert False

        random_actor = copy.deepcopy(self.actor)

        #self.actor = SigNet(D, MEMORY**2, hidden_size= D*2, output_size= 2, batch_size= c_batch_size*HORIZON)
        #self.actor = SplitNet(self.D, MEMORY**2, hidden_size= self.D*2, output_size= 2, batch_size= c_batch_size*HORIZON)
        #self.actor = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, hidden_size= hid_size, output_size= 2, output= 'sig', batch_size= c_batch_size*HORIZON, reg= reg)
        #self.critic = EuclidNet(D, MEMORY**2, hidden_size= D*2, output_size= 1, batch_size= c_batch_size*HORIZON)
        #self.critic = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, hidden_size= hid_size, output_size= 1, output= 'euclid', batch_size= c_batch_size*HORIZON, reg= reg)

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

        mean_train_rewards = np.zeros(num_iters)
        mean_val_rewards = np.zeros(num_iters)
        mean_rand_rewards = np.zeros(num_iters)

        cum_train_rewards = np.zeros(num_iters)
        cum_val_rewards = np.zeros(num_iters)
        cum_rand_rewards = np.zeros(num_iters)

        total_cum_train_rewards = np.zeros(num_iters)
        total_cum_val_rewards = np.zeros(num_iters)
        total_cum_rand_rewards = np.zeros(num_iters)

        for it in range(num_iters):
            print("ITERATION: ", it)

            #trainpoints = np.random.choice(range(len(self.env.train_sample)), epoch_tsample)

            train_rewards = []
            rand_rewards = []

            for d in range(train_size):
                rewards = self.batch_cacla(d, c_batch_size, num_epochs, mode= 'train')
                train_rewards.append(rewards)

                rrewards, racts = self.simulate(agent=random_actor, mode='train',
                                               imgIdx=d)
                rand_rewards.append(rrewards)

            #valpoints = np.random.choice(range(len(self.env.val_sample)), epoch_vsample)

            val_rewards = []
            for d in range(val_size):
                rewards = self.batch_cacla(d, c_batch_size, num_epochs, mode= 'val')
                val_rewards.append(rewards)

            cum_train_rewards[it] = np.array(train_rewards).sum()
            cum_val_rewards[it] = np.array(val_rewards).sum()
            cum_rand_rewards[it] = np.array(rand_rewards).sum()

            #total_cum_train_rewards[it] = total_cum_train_rewards.sum() +  np.array(train_rewards).sum()
            #total_cum_val_rewards[it] = total_cum_val_rewards.sum() + np.array(val_rewards).sum()
            total_cum_train_rewards[it] = cum_train_rewards[:it].mean()
            total_cum_val_rewards[it] = cum_val_rewards[:it].mean()
            total_cum_rand_rewards[it] = cum_rand_rewards[:it].mean()

            mean_train_rewards[it] = np.array(train_rewards).mean()
            mean_val_rewards[it] = np.array(val_rewards).mean()
            mean_rand_rewards[it] = np.array(rand_rewards).mean()

        if save == True:
            folder_name = time.strftime("%d-%m-%Y")+'_'+(time.strftime("%H-%M-%S"))
            os.mkdir('models/'+folder_name)

            act_params = lasagne.layers.get_all_param_values(self.actor.network)
            crit_params = lasagne.layers.get_all_param_values(self.actor.network)

            epochs = str(num_iters + start_iter)

            actorid = "ACTOR-D-"+str(D)+"-rad-"+str(RADIUS)+"R_THRESH"+str(R_THRESH)+"-mem-"+str(MEMORY)+"-reg-"+str(reg)+"-eps-"+str(EPSILON)+"-ts-"+str(train_size)+"-vs-"+str(val_size)+"-epochs-"+epochs+"-id-"+self.actor.return_id()
            criticid = "CRITIC-D-"+str(D)+"-rad-"+str(RADIUS)+"R_THRESH"+str(R_THRESH)+"-mem-"+str(MEMORY)+"-reg-"+str(reg)+"-eps-"+str(EPSILON)+"-ts-"+str(train_size)+"-vs-"+str(val_size)+"-epochs-"+epochs+"-id-"+self.critic.return_id()

            np.savez('models/'+folder_name+'/mean_train_rewards', mean_train_rewards)
            np.savez('models/'+folder_name+'/mean_val_rewards', mean_val_rewards)
            np.savez('models/'+folder_name+'/mean_rand_rewards', mean_rand_rewards)

            np.savez('models/'+folder_name+'/cum_train_rewards', cum_train_rewards)
            np.savez('models/'+folder_name+'/cum_val_rewards', cum_val_rewards)
            np.savez('models/'+folder_name+'/cum_rand_rewards', cum_rand_rewards)

            np.savez('models/'+folder_name+'/total_cum_train_rewards', total_cum_train_rewards)
            np.savez('models/'+folder_name+'/total_cum_val_rewards', total_cum_val_rewards)
            np.savez('models/'+folder_name+'/total_cum_rand_rewards', total_cum_rand_rewards)

            np.savez('models/'+folder_name+'/'+actorid, act_params)
            np.savez('models/'+folder_name+'/'+criticid, crit_params)

            np.savez('models/'+folder_name+'/'+'train_set', self.env.train_sample)
            np.savez('models/'+folder_name+'/'+'val_set', self.env.val_sample)

        #Plot reward metrics

        if display:
            f, (ax1, ax2, ax3) = plt.subplots(3, 1)
            ax1.plot(mean_train_rewards, c= 'b')
            ax1.plot(mean_val_rewards, c= 'r')
            ax1.plot(mean_rand_rewards, c= 'g')

            ax2.plot(cum_train_rewards, c= 'b')
            ax2.plot(cum_val_rewards, c= 'r')
            ax2.plot(cum_val_rewards, c= 'g')

            ax3.plot(total_cum_train_rewards, c= 'b')
            ax3.plot(total_cum_val_rewards, c= 'r')
            ax3.plot(total_cum_val_rewards, c= 'g')

            plt.show()


print "RADIUS: ", RADIUS

#CAT_COUNT_TRAIN = {'Action':5, 'Object':5}
#CAT_COUNT_VAL = {'Action':5, 'Object':5}

#CAT_COUNT_TRAIN = {'Random':5, 'OutdoorManMade':5, 'Social':5}
#CAT_COUNT_VAL = {'Action':5, 'Object':5, 'Social':5}

categories = ['Action','Affective','Art','BlackWhite','Cartoon','Fractal','Indoor','Inverted',\
              'Jumbled','LineDrawing','LowResolution','Noisy','Object','OutdoorManMade','OutdoorNatural',
              'Pattern','Random','Satelite','Sketch','Social']

#CAT_COUNT_TRAIN = {'Action':30, 'Object':30, 'Social':30, 'OutdoorManMade':30, 'Affective':30}
#CAT_COUNT_VAL = {'Action':10, 'Object':10, 'Social':10, 'OutdoorManMade':10, 'Affective':10}
CAT_COUNT_TRAIN = {'Action':15}
CAT_COUNT_VAL = {'Action':5}

CAT_COUNT_PCA = {key:10 for key in categories}

#loadRewardPlots('models/26-02-2016_14-04-45')
#loadRewardPlots('models/26-02-2016_12-44-23')
#loadRewardPlots('models/26-02-2016_12-19-02') #Convolutional Network
#loadRewardPlots('models/26-02-2016_17-34-16_PCA')
#loadRewardPlots('models/27-02-2016_13-48-50_PCA200')
#loadRewardPlots('models/27-02-2016_14-20-08_PCA200') #The qualitative results look good.
#loadRewardPlots('models/27-02-2016_17-20-20')
#loadRewardPlots('models/27-02-2016_18-09-17_PCA200')
#loadRewardPlots('models/27-02-2016_19-30-45_PCA400')
#loadRewardPlots('models/27-02-2016_21-59-34')
#loadRewardPlots('models/26-02-2016_12-19-02')
#loadRewardPlots('models/28-02-2016_05-57-09')

def train_an_agent(phi, actor, critic, num_iters= 20, save= False, display= True):
    S_train = exactCAT(CAT_COUNT_TRAIN, 'train', size= (256, 256), asgray= True)
    S_val = exactCAT(CAT_COUNT_VAL, 'val', size= (256, 256), asgray= True)

    env = Environment(S_train, S_val)
    age = Agent(env, actor, critic, phi= phi)
    age.train(None, num_iters= num_iters, c_batch_size = 10, save= save, display= display)

    return age

def load_an_agent(folder, name):
    list_params = np.load('models/'+folder+'/'+name)
    list_params = list_params.items()[0][1]

    S_train = exactCAT(CAT_COUNT_TRAIN, 'train', size= (256, 256), asgray= True)
    S_val = exactCAT(CAT_COUNT_VAL, 'val', size= (256, 256), asgray= True)
    env = Environment(S_train, S_val)

    c_batch_size= 10

    arc = name.split('-')[-1].split('.')[0]

    if arc == 'sig':
        phi = pcaExtract((200,))
        D = phi.D[0]
        hid_size = int(np.prod(D)*(4.0/3))
        S_PCA = exactCAT(CAT_COUNT_PCA, 'val', size= (256, 256), asgray= True)
        phi.train(S_PCA, 50)

        actor = SigNet(D, MEMORY**2, hidden_size= hid_size, output_size= 2, batch_size= c_batch_size*HORIZON)
        age = Agent(env, actor= actor, phi = phi)

    elif arc == 'split':
        phi = pcaExtract((200,))
        D = phi.D[0]
        hid_size = int(np.prod(D)*(4.0/3))
        age = Agent(env, phi = phi)
        S_PCA = exactCAT(CAT_COUNT_PCA, 'val', size= (256, 256), asgray= True)
        phi.train(S_PCA, 50)

        age.actor = SplitNet(D, MEMORY**2, hidden_size= hid_size, output= 'sig', output_size= 2, batch_size= c_batch_size*HORIZON)

    elif arc == 'csn':
        phi = IdentityExtract((1, 2*RADIUS, 2*RADIUS))
        D = phi.D
        hid_size = int(np.prod(D)*(4.0/3))

        reg = 1e-2
        actor = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, hidden_size= hid_size, output_size= 2, output= 'sig', batch_size= c_batch_size*HORIZON, reg= reg)
        age = Agent(env, actor= actor, phi = phi)
        #age.critic = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, hidden_size= hid_size, output_size= 1, output= 'euclid', batch_size= c_batch_size*HORIZON, reg= reg)
    else:
        assert False

    lasagne.layers.set_all_param_values(age.actor.network, list_params)

    return age

#age = load_an_agent('27-02-2016_13-48-50_PCA200', 'ACTOR-D-200-rad-10R_THRESH150-mem-6-reg-0.01-eps-45-ts-150-vs-50-epochs-20-id-sig.npz')

phi = IdentityExtract((1, 2*RADIUS, 2*RADIUS))
c_batch_size = 10
reg = 0.0
hid_size= 200
actor = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, pre_conv= False, num_hiddense= 2, hidden_size= hid_size, output_size= 2, output= 'sig', batch_size= c_batch_size*HORIZON, reg= reg)
critic = ConvSplitNet(2*RADIUS, 2*RADIUS, 1, MEMORY**2, pre_conv= False, num_hiddense= 2, hidden_size= hid_size, output_size= 1, output= 'euclid', batch_size= c_batch_size*HORIZON, reg= reg)
age = train_an_agent(phi, actor, critic, num_iters= 10, save= True, display= True)

if True:
    #trainpoints = np.random.choice(range(len(age.env.train_sample)), 9)
    train_rewards = []
    for d in range(len(age.env.train_sample)):
        age.env.update(d, mode= 'train')
        rewards, actions = age.simulate(mode= 'train', display= 'fixmap', verbose= True)

        A = np.array(actions)
        A = A * [age.env.M, age.env.N]

        plt.imshow(-1 * age.env.currTrial.image, cmap = 'Greys')
        plt.plot(A[:,0], A[:,1])

        plt.show()

        halt= True
else:
    #valpoints = np.random.choice(range(len(age.env.val_sample)), 3)
    train_rewards = []
    for d in range(len(age.env.val_sample)):
        age.env.update(d, mode= 'val')
        rewards, actions = age.simulate(mode= 'val', display= 'img', verbose= True)
