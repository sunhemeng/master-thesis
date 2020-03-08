import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
from scipy import special
from scipy.stats import rayleigh
from itertools import product
#import Sim_Optimal_Offloading_V2
from DDPG import DDPGAgent
from OUNoise import OUNoise
from Sim_Optimal_Offloading_V2 import Optimal_Offloading

'''
Simulation Environment for partitioned workload. Needs to include the DDPGAgent and OUNoise. 
'''

class Simulation:
    def __init__(self, number_of_servers=3, number_of_users=1, historic_time=4, snr_set=10, csi=0, channel=0.9):  # initialize number of servers, number of users,
                                                                                                                  # historic timeslots, avg_snr, csi knowledge, channel correlation
        # define Simulation parameters
        self.S = number_of_servers  # number of servers
        self.N = number_of_users  # number of users
        self.W = historic_time     # historic time slots
        self.features = 2  # number of feature(SNR and workload in state)
        self.action = self.compute_action_space()  # action set
        self.server_selection_size = len(self.action)
        self.CSI = csi        # 0 = information of previous time slot, 1 = perfect CSI
        self.p = channel      # channel correlation between time slots

        print(self.action)

        # Communication & Computation constants
        # self.area_width = 50
        # self.area_length = 50
        # self.poisson_variable = np.random.poisson(0.5 * self.area_width, 3)
        # self.A = self.area_length * self.area_width  # area in square meter
        # self.B = 5 * 10 ** 6  # bandwidth
        # self.F = 2.4 * 10 ** 9  # carrier frequency
        # self.r = []
        # self.r = [110, 110, 110, 20, 20, 20, 20, 20, 20, 20, 20, 20]  # distance between UE and server
        # self.No_dBm = -174  # noise power level in dbm/Hz
        # self.No_dB = self.No_dBm - 30
        self.TS = 0.025 * 10 ** -3  # symbol time 0.025 *10**-3
        # self.P_dBm = 20  # in dbm
        # self.P_dB = self.P_dBm - 30
        self.pi = 320  # bits
        self.co = 24 * 10 ** 6  # total workload in cycles 24
        self.f = 3 * 10 ** 9  # computation power
        self.lam = 3 * 10 ** 6  # poisson arriving rate
        self.xi = -0.0214  # shape parameter
        self.sigma = 3.4955 * 10 ** 6  # scale parameter
        self.d = 2.0384 * 10 ** 7  # threshold
        self.T = 0.025  # delay tolerance in s 0.025
        self.eps_max = 0.001
        self.threshold_computation = 0.999  # computation threshold
        self.comp_correlation = 4*10**6    # computation correlation

        # Feedback
        self.reward = 0
        self.error = 0
        # self.rnd_channel = np.random.seed(2236391566)

        #print(np.random.get_state())

        self.channel_type = self.channel_type_name()        # Create Channel type string for saving data in right path
        print(self.channel_type)

        self.h = []             # channel gain
        for i in range(self.S):
            self.h = np.append(self.h, 1)
        #     self.h = np.append(self.h, 1)


        self.channel_sequence = self.create_random_channel()        # create fixed channel sequence
        print('random:', self.channel_sequence[0][0])

        ### Initialize SNR array ###
        self.SNR_s = []                                 # initialize SNR of channel
        self.SNR_avg = [np.power(10, snr_set/10)]*self.S
        self.SNR = []
        for i in range(self.S):
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i])         # set average snr 
        for i in range(self.W):                                         # initialize whole array
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)
        self.SNR = self.SNR.reshape((self.W, self.S))
        print(self.SNR)

        ### Initiliaze previous task sizes ###
        self.tasks_prev = []        # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)      # initialize whole array
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI), self.S)
        print(self.tasks_prev)

        # Initialize state, concatenate snr + previous tasks sizes dependent on knowledge of environment
        if self.CSI == 1:       # perfect CSI
            self.state = np.log10(self.SNR) / 10    # initialize state
        else:                   # outdated CSI
            self.state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)    # initialize state

        print(self.state)

    def reset(self):
        '''
        resets reward, error variables to zero. Sets SNR and workload back.
        :return:
        '''
        # Feedback
        self.reward = 0
        self.error = 0

        self.h = []             # channel gain
        for i in range(self.S):
            self.h = np.append(self.h, 1)

        ### Initialize SNR array ###
        self.SNR_s = []  # initialize SNR of channel
        self.SNR = []
        for i in range(self.S):
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i])
        for i in range(self.W):
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)
        self.SNR = self.SNR.reshape((self.W, self.S))

        self.tasks_prev = []  # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI, self.S))
        # print(self.tasks_prev)

        ### Initialize state, concatenate snr + previous tasks sizes ###
        if self.CSI == 1:       # perfect CSI
            self.state = np.log10(self.SNR) / 10
        else:                   # outdated CSI
            self.state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)

    def channel_type_name(self):

        if self.p == 1:
            channel = 'StaticChannel'
        else:
            channel = 'FadingChannel'

        if self.CSI == 1:
            channel = channel + '_PerfectCSI'
        else:
            channel = channel + '_OutdatedCSI'
        return channel

    def create_random_channel(self):
        '''
        create channel sequence h^
        :return: returns h^ for every server with length 1000
        '''
        h_c = []
        for c in range(self.S):     # for every channel
            h_c.append([])

        for s in range(self.S):
            np.random.seed(s)
            for i in range(1000):           # length of channel
                h_c[s].append(complex(np.random.randn(), np.random.randn()) / np.sqrt(2))
        return h_c

    def compute_action_space(self):
        # l = list(product(range(2), repeat=3))
        dck = [0, 1/6, 1/3, 2/3, 1]
        l = list(product(dck, repeat=self.S))
        # l = list(product([0, 4, 8, 16, 24], repeat=self.S))
        # act_all = np.asarray(l)
        l = np.round(np.asarray(l), 4)
        act_all = l[1:]
        # print(act_all)
        act_space = []
        for i in range(len(act_all)):
            if np.sum(act_all[i]) < 0.99:
                continue
            if round(np.sum(act_all[i]) - 0.01 + 0.5) == 1:
                act_space.append(act_all[i])
        #print(np.asarray(act_space))
        return np.asarray(act_space)

    def compute_action_space_V1(self):
        '''
        :return: action space regarding server selection
        '''
        l = list(product(range(2), repeat=self.S))
        action_space = np.asarray(l)
        action_space = action_space[1:]
        return action_space

    def total_error(self, n, c):
        '''
        computes overall error depending on set parameters and arguments n and c
        :param n: blocklength
        :param c: workload array
        :return: overall error
        '''
        t1 = self.TS * n
        ck = c * self.co
        t2 = self.T - t1
        # Communication error
        r = self.pi / n  # coding rate
        shannon = np.log2(1 + self.SNR[-1])  # shannon
        channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
        Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
        comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
        # print([t2 - ((ck + self.d) + self.tasks_prev - self.time_slot_const) / self.f, 0])

        # Computation error
        if self.CSI==1:
            comp_depend = 0
        else:
            comp_depend = self.tasks_prev[-1] - self.comp_correlation
            comp_depend[comp_depend < 0] = 0

        used = np.clip(ck, 0, 1)
        m = t2 - (((ck + self.d) + comp_depend) / self.f)
        m[m < 0] = 0
        # print('m:', m)
        comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                        -1 / self.xi)

        total_k = used * (comm_error + comp_err - (
                        comm_error * comp_err))  # compute error of single channel + computation

        # for i in range(self.S):  # total error
        #     if total_k[i] > 0.01:
        #         total_k[i] = 1

        total = 1 - np.prod(1 - total_k)        # compute overall error

        # Update workload array
        self.tasks_prev = np.append(self.tasks_prev[1:], ck)
        self.tasks_prev = self.tasks_prev.reshape((self.W -1 + self.CSI, self.S))
        return total

    def update_channel_state(self):
        '''
        Update channel using total random channel variable
        :return:
        '''
        SNR = []
        for i in range(self.S):          # compute new SNR for every channel
            h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
            hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
            SNR = np.append(SNR, self.SNR_avg[i] * abs(np.square(hdl)))
            self.h[i] = hdl
        self.SNR = np.append(self.SNR[1:], SNR)
        self.SNR = self.SNR.reshape((self.W, self.S))

    def update_channel_state_sequence(self, count):
        '''
        Updates channel based on pseudo-random channel sequence
        :param count: Cycle counter
        :return: update of SNR array
        '''
        SNR = []
        for i in range(self.S):          # compute new SNR for every channel
            h_bar = self.channel_sequence[i][count]     # using fixed channel sequence
            hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
            SNR = np.append(SNR, self.SNR_avg[i] * abs(np.square(hdl)))
            self.h[i] = hdl
        self.SNR = np.append(self.SNR[1:], SNR)
        self.SNR = self.SNR.reshape((self.W, self.S))

    def create_state(self):
        '''
        Creates state dependent on buffer and CSI
        :return: state (m+1)
        '''

        # division of 10 is for regulating input values in same range (can be changed)
        if self.CSI == 1:       # perfect CSI
            state = np.log10(self.SNR) / 10
        else:                   # outdated CSI
            state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)
        # state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)
        #print(state)
        return state

    def compute_rewards(self, error_probability):
        '''
        Computes reward depending on error
        :param error_probability:
        :return: reward r
        '''
        e = error_probability

        if e == 0:
            self.reward = 1
            return

        # if e == 1:
        #     self.reward = -1
        #     return

        self.reward = - np.log10(e)/100     # use 100 as factor (can be changed to best outcome)

    def Assign_Cores(self, actions, count, train):
        '''
        :param actions: workload index
        :param count: Cycle counter for pseudo-random channel sequence
        :param train: Training or testing (1 for training)=> which channel update should be taken?
        :return: next state, reward, error
        '''

        action_blkl = int(round((actions[0]) * (self.T/self.TS)))   # map value to blocklength
        action_selection = self.action[int(round(actions[1]*(len(self.action) -1)))]    # map value to index in workload array
        self.reward = 0                 # set reward to 0
        non_zeros = np.count_nonzero(action_selection)        # check if any server is choosen

        if non_zeros > 0:
            #blkl = self.calculate_blocklength(ac)
            total_error = self.total_error(action_blkl, action_selection)      # computes the error probability for optimal t
            self.compute_rewards(total_error)         # assigns reward
            if total_error < 10**-100:
                total_error = 10**-100
            self.error += - np.log10(total_error)

        if train == 1:          # training or testing?
            self.update_channel_state()            # update channel SNR for next state
        else:
            self.update_channel_state_sequence(count)
        self.state = self.create_state()       # updates next state
        #print(self.state)
        return self.state, self.reward, total_error

num_of_servers_a = [2, 3, 4]      # set number of servers
hist_timeslots = 4          # number of historical timeslots
avg_SNR = 10                # avg SNR
#rho_a = [0.2, 0.6]
# rho = 0.9
num_of_servers = 2

for r in num_of_servers_a:
    # rnd_channel = np.random.seed(22363)
    # rand = random.seed(1444)
    # Initialize Simulation Environment
    sim_env = Simulation(number_of_servers=r, number_of_users=1, historic_time=hist_timeslots,
                         snr_set=avg_SNR, csi=0, channel=0.9)  # number_of_server = 3, number_of_users = 1, historic time slots, avgSNR, perfCSI?(yes=0, no=1), channel correlation

    episodes = 400          # number of episodes
    training = 1000         # training time for each episode
    testing = 1000          # testing time for each episode

    # Q-Network
    state_size = ((sim_env.features-sim_env.CSI) * (sim_env.W - 1 + sim_env.CSI) * sim_env.S)       # Calculate state size
    action_size = 2             # set action size ( 2 for blocklength and workload index)
    batch_size = 512            # set batch size
       
     # initialize DDPG agent and noise process, set decay factor(gamma), learning rates, tau(soft copy factor), batch size
    QN = DDPGAgent(state_size=state_size, action_size=2, gamma=0.0, learning_rate_actor=0.00005, learning_rate_critic=0.0001, tau=0.001, batch_size=batch_size, action_max=[1, 1])
    #  state_size, action size, discount factor(gamma), learning_rate_actor, learning_rate_critic, soft_copy factor(tau), batch size, maximum value for actions
    # set decay_period in OUNoise module directly
    Noise = OUNoise(action_space=action_size,  mu=np.asarray([0.0, 0.0]), theta=np.asarray([0.5, 0.5]), max_sigma=np.asarray([0.8, 0.8]), min_sigma=np.asarray([0.1, 0.05]), action_max=[1, 1], action_min=[0.001, 0])
    # action_size, mean reversion level(mu), mean reversion speed(theta), random factor influence (max and min), maximum action value (1,1), minimum action value

    error_avg = []      # declare avg error array
    error = []          # declare absolute error array
    states = sim_env.state
    ee = 0
    for e in range(episodes):
        sim_env.reset()
        for k in range(training):
            train = 1
            states = np.reshape(states, [1, state_size])        # reshape state array to vector for network
            action = QN.policy_action(states)                   # decide on action based on state
            action = Noise.get_action(action)                   # add noise to action
            next_state, rewards, overall_err = sim_env.Assign_Cores(action, k, train)    # give action to environment and return next state, reward and error
            ee += overall_err
            next_state = np.reshape(next_state, [1, state_size])            # reshape next state
            QN.remember(states, action, rewards, next_state)                # save state in replay memory
            states = next_state                         # state = next state
            if len(QN.memory) > batch_size:             #if memory is larger than batch size then train
                QN.replay(batch_size)

        print(ee/training)
        QN.grad_avg = 0
        #print(sim_env.eps_max)
        sim_env.reset()
        for u in range(testing):
            train = 0           # no train => fixed channel sequence in environment
            states = np.reshape(states, [1, state_size])        # reshape state array to vector for network
            action = np.clip(QN.policy_action(states), [0.001, 0], [1, 1])      # decide on action based on state
            # print('##################test#######################')
            print('action_S{}:'.format(sim_env.S), action)
            next_state, rewards, overall_err = sim_env.Assign_Cores(action, u, train)   # give action to environment and return next state, reward and error
            # print('reward:', rewards)
            # print('error:', overall_err)
            # print(rewards)
            error = np.append(error, overall_err)               # add error to array
            next_state = np.reshape(next_state, [1, state_size])        # reshape next state
            states = next_state             # state = next state
        QN.grad_avg = 0
        print(e)
        print(sim_env.error/testing)
        error_avg = np.append(error_avg, np.power(10, -sim_env.error / testing))        # save average error
        # print('Nach Test:', sim_env.SNR)

    grad_overall = np.asarray(QN.grad_a).reshape((int(len(QN.grad_a) / QN.action_size), QN.action_size))        # reshape action gradient vector
    dir = sim_env.channel_type
    # set parameter values for saving data
    parameters = '_DDPG_S{}_rho{}_SNR{}_PS{}_lr{}_df{}_W{}_sigOU{}_thetaOU{}_Critic_V2'.format(sim_env.S, sim_env.p, sim_env.SNR_avg[0], sim_env.pi, QN.learning_rate_actor,
                    QN.gamma, sim_env.W, Noise.max_sigma[0], Noise.theta[0])
    # save Average Error
    np.savetxt(dir + '/Error' + parameters + '.csv', np.transpose(error_avg), header='Error[sum(-log10(e))]',
               fmt='0%30.28f')
    # Save Absolute Error
    np.savetxt(dir + '/AbsError' + parameters + '.csv', np.transpose(error), header='Absolute Error of every Cycle',
               fmt='0%30.28f')
    # Save Action Gradient for each batch
    np.savetxt(dir + '/AvgGradient' + parameters + '.csv', grad_overall,
               header='Average_Gradient of every Episode', fmt='%30.28f')
    # Save loss of critic network
    np.savetxt(dir + '/AvgLoss_Critic' + parameters + '.csv', QN.critic_loss_a,
               header='Average_Loss of every Training from Critic Network', fmt='%30.28f')
    QN.save_weights(dir + '/', parameters)
