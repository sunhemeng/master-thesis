import numpy as np
'''
class for noise process
'''
class OUNoise(object):
    '''
    Initialize Ornstein Uhlenbeck Process
    '''
    def __init__(self, action_space=2, mu=np.asarray([50, 0.35]), theta=np.asarray([1, 0.5]),
                 max_sigma=np.asarray([750, 5]), min_sigma=np.asarray([100, 1]), decay_period=100000,
                 action_max=[2000, 2], action_min=[0, 0]):
        self.mu = mu        # mean reversion level
        self.theta = theta      # reversion speed
        self.sigma = max_sigma      # factor of random variable
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period        # steps for reducing random impact
        self.action_dim = action_space          # dimension of action space
        self.low = action_min                   # minimum of each action
        self.high = action_max                  # maximum of each action
        self.t = 1
        self.exploration = 1
        self.reset()


    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state          # set previous noise
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)     # calculate dx
        # print('dx:', dx)
        self.state = x + dx  # next noise
        #print('noise')
        #print(self.state)
        return self.state

    def get_action(self, action):
        ou_state = self.evolve_state()      # get noise
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)  # new sigma
        self.t += 1     # iterate
        # print('noise:', ou_state)
        return np.clip(action + ou_state, self.low, self.high) # do not exceed max and min action

    # def get_action_noiseDependent(self, action):
        ### get last action ###
        # la = 1 - np.sum(action[1:])
        # action = np.append(action, la)
        # print('bef_norm:', action)
       # ou_state = self.evolve_state()
       # self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, self.t / self.decay_period)
       # self.t += 1
        # print('noise:', ou_state)
       # ou_state_ac = np.clip(action + ou_state, self.low, self.high)
       # if np.count_nonzero(ou_state_ac[1:]) == 0:
       #     ou_state_ac[1:] = [1/(self.action_dim-1)]*(self.action_dim-1)
        # print('clipped:', ou_state_ac)
        ### normalize workload actions ###
       # ou_state_norm = (ou_state_ac[1:] / np.sum(ou_state_ac[1:]))
       # diff = 1 - sum(ou_state_norm)
       # if diff != 0:
       #     r = np.random.randint(self.action_dim-1)
       #     ou_state_norm[r] = ou_state_norm[r] - diff
        # print('action + noise:', ou_state_norm)

       # return np.append(ou_state_ac[0], ou_state_norm)

    # def evolve_state_norm(self):
    #     x = self.state
    #     dx = self.theta * (self.mu - x) + self.sigma * self.norm_random()
    #     self.state = x + dx
    #     #print('noise')
    #     #print(self.state)
    #     return self.state
    #
    # def norm_random(self):
    #     noise = np.random.randn(self.action_dim)
    #     noise_sum = sum(noise)
    #     noise_n = (noise / noise_sum) - 1 / (len(noise))
    #     diff = sum(noise_n)
    #     if diff != 0:
    #         r = [np.random.randint(self.action_dim)]
    #         noise_n[r] = noise_n[r] - diff
    #     return noise_n
