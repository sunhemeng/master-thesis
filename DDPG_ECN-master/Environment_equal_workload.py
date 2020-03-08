import matplotlib.pyplot as plt
from random import Random
import numpy as np
from collections import deque
from scipy import special
from scipy.stats import rayleigh
from itertools import product
#import Sim_Optimal_Offloading_V2
from DDPG import DDPGAgent
from OUNoise import OUNoise
from Sim_Optimal_Offloading_V2 import Optimal_Offloading


class Simulation:
    def __init__(self, number_of_servers=3, number_of_users=1, historic_time=4, snr_set=10, csi=0, channel=0.9):
        # define Simulation parameters
        self.S = number_of_servers  # number of servers
        self.N = number_of_users  # number of users
        self.W = historic_time
        self.features = 2  # number of feature
        self.action = self.compute_action_space()  # action set
        self.server_selection_size = len(self.action)
        self.CSI = csi  # 0 = information of previous time slot, 1 = perfect CSI
        self.p = channel  # channel correlation between time slots

        print(self.action)

        # Communication & Computation constants
        self.area_width = 50
        self.area_length = 50
        self.poisson_variable = np.random.poisson(0.5 * self.area_width, 3)
        self.A = self.area_length * self.area_width  # area in square meter
        self.B = 5 * 10 ** 6  # bandwidth
        self.F = 2.4 * 10 ** 9  # carrier frequency
        self.r = []
        self.r = [110, 110, 110, 20, 20, 20, 20, 20, 20, 20, 20, 20]  # distance between UE and server
        self.No_dBm = -174  # noise power level in dbm/Hz
        self.No_dB = self.No_dBm - 30
        self.TS = 0.025 * 10 ** -3  # symbol time 0.025 *10**-3
        self.P_dBm = 20  # in dbm
        self.P_dB = self.P_dBm - 30
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
        self.comp_correlation = 4 * 10 ** 6  # computation correlation
        # evaluation variables
        self.comm_error = 0
        self.comp_error = 0
        # Feedback
        self.reward = 0
        self.error = 0

        # print(np.random.get_state())

        self.channel_type = self.channel_type_name()
        print(self.channel_type)

        self.h = []  # channel gain
        for i in range(self.S):
            self.h = np.append(self.h, 1)

        ### Initialize SNR array ###
        self.SNR_s = []  # initialize SNR of channel
        self.SNR_avg = [np.power(10, snr_set / 10)] * self.S
        self.SNR = []
        for i in range(self.S):
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i])
        for i in range(self.W):
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)
        self.SNR = self.SNR.reshape((self.W, self.S))
        print(self.SNR)

        ### Initiliaze previous task sizes ###
        self.tasks_prev = []  # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI), self.S)
        print(self.tasks_prev)

        ### Initialize state, concatenate snr + previous tasks sizes ###
        if self.CSI == 1:  # perfect CSI
            self.state = np.log10(self.SNR)
        else:  # outdated CSI
            self.state = np.concatenate((np.log10(self.SNR[:-1]) / 10, self.tasks_prev / (10 ** 9)), axis=0)  # initialize state

        print(self.state)

    def reset(self):
        '''
        resets reward, error variables to zero. Sets SNR and workload back.
        :return:
        '''
        # Feedback
        self.reward = 0
        self.error = 0

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
            self.state = np.log10(self.SNR)
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

    def compute_action_space(self):
        '''
        :return: action space regarding server selection
        '''
        l = list(product(range(2), repeat=self.S))
        action_space = np.asarray(l)
        action_space = action_space[1:]
        return action_space

    def total_error(self, n, a):
        '''
        computes overall error depending on set parameters and arguments n and c
        :param n: blocklength
        :param a: server selection array
        :return: overall error
        '''
        t1 = self.TS * n
        ck = self.co / np.count_nonzero(a)   # assgin equal workload to every selected server
        t2 = self.T - t1
        r = self.pi / n  # coding rate

        # Communication
        shannon = np.log2(1 + self.SNR[-1])  # shannon
        channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
        Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
        comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
        # print([t2 - ((ck + self.d) + self.tasks_prev - self.time_slot_const) / self.f, 0])

        #Computation
        if self.CSI == 1:
            comp_depend = 0
        else:
            comp_depend = self.tasks_prev[-1] - self.comp_correlation
            comp_depend[comp_depend < 0] = 0
        m = t2 - ((((ck*a) + self.d) + comp_depend) / self.f)
        m[m < 0] = 0
        # print('m:', m)
        comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                -1 / self.xi)

        # Link k error
        total_k = a * (comm_error + comp_err - (
                comm_error * comp_err))  # compute error of single channel + computation

        # for i in range(self.S):  # total error
        #     if total_k[i] > 0.01:
        #         total_k[i] = 1

        total = 1 - np.prod(1 - total_k)        # overall error
        self.tasks_prev = np.append(self.tasks_prev[1:], a * ck)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI, self.S))

        # print('total_k:', total_k)
        # print('SNR:', self.SNR[-1])
        # print('comm:', comm_error)
        # print('comp:', comp_err)
        # print('total:', total)
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

    def create_state(self):
        '''
        Creates state dependent on buffer and CSI
        :return: state (m+1)
        '''
        if self.CSI == 1:       # perfect CSI
            state = np.log10(self.SNR)/10
            # state = self.SNR
        else:                   # outdated CSI
            state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)
        # state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**9)), axis=0)
        #print(state)
        return state

    def compute_rewards(self, error_probability, used):
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

        self.reward = - np.log10(e)/100

    def Assign_Cores(self, actions):
        '''
        :param actions: actions include blocklength and index for server selection array
        :return: next state, reward, error
        '''
        action_blkl = int(round((actions[0]) * (self.T/self.TS)))       # take blocklength
        action_selection = self.action[int(round(actions[1]*(len(self.action) -1)))] # select servers
        self.reward = 0                 # set reward to 0
        non_zeros = np.count_nonzero(action_selection)        # check if any server is choosen

        if non_zeros > 0:
            #blkl = self.calculate_blocklength(ac)
            total_error = self.total_error(action_blkl, action_selection)      # computes the error probability for optimal t
            self.compute_rewards(total_error, action_selection)         # assigns reward
            if total_error < 10**-100:
                total_error = 10**-100
            self.error += - np.log10(total_error)

        self.update_channel_state()            # update channel SNR for next state
        self.state = self.create_state()       # updates next state
        #print(self.state)
        return self.state, self.reward, total_error


loop_start = 6
loop_end = 10

W = 1
SNR = 10
OU_theta = 0.3
OU_sigma = 0.8
df = 0.0
lr = 0.0001

for r in range(loop_start, loop_end + 1):
    # Initialize Simulation Environment
    sim_env = Simulation(number_of_servers=r, number_of_users=1, historic_time=W,
                         snr_set=SNR, csi=1, channel=1)  # number_of_server = 3, number_of_users = 1, historic time slots, avgSNR

    episodes = 100
    training = 1000
    testing = 1000
    testing_comps = 5000

    success_ratio = []
    # Q-Network
    state_size = ((sim_env.features-sim_env.CSI) * (sim_env.W - 1 + sim_env.CSI) * sim_env.S)
    action_size = 2
    batch_size = 128

    QN = DDPGAgent(state_size=state_size, action_size=2, gamma=df, learning_rate_actor=lr/2,
                   learning_rate_critic=lr, tau=0.001, batch_size=batch_size, action_max=[1, 1])

    # Noise
    mu = np.concatenate(([0], [0]))
    theta = np.concatenate(([OU_theta - 0.2], [OU_theta]))
    max_sigma = np.concatenate(([OU_sigma - 0.2], [OU_sigma]))
    min_sigma = np.concatenate(([0.001], [0.001]))
    action_max = [1] * 2
    action_min = np.concatenate(([0.01], [0]))
    Noise = OUNoise(action_space=2, mu=mu, theta=theta, max_sigma=max_sigma, min_sigma=min_sigma,
                    action_max=action_max, action_min=action_min)

    grads_overall = []
    error_avg = []
    error = []
    ee = 0
    states = sim_env.state
    for e in range(episodes):
        sim_env.reset()
        for k in range(training):
            states = np.reshape(states, [1, state_size])
            # print('states:', states)
            #action = QN.policy_action(states)
            # print('#########train_cycle start############')
            action = QN.policy_action(states)
            # print('act_before:', action)
            # print(action)
            action = Noise.get_action(action)
            # print('act_after:', action)
            # print(action)
            next_state, rewards, overall_err = sim_env.Assign_Cores(action)
            # print('reward:', rewards)
            # print('error:', overall_err)
            ee += overall_err
            # print(rewards)
            next_state = np.reshape(next_state, [1, state_size])
            QN.remember(states, action, rewards, next_state)
            states = next_state
            if len(QN.memory) > batch_size:
                QN.replay(batch_size)
        # grads_overall = np.append(grads_overall, QN.grad_avg/training)
        QN.grad_avg = 0
        sim_env.reset()
        for u in range(testing):
            states = np.reshape(states, [1, state_size])
            action = np.clip(QN.policy_action(states), [0.001, 0], [1, 1])
            # print('##################test#######################')
            print('action:', action)
            next_state, rewards, overall_err = sim_env.Assign_Cores(action)
            # print('reward:', rewards)
            # print('error:', overall_err)
            # print(rewards)
            error = np.append(error, overall_err)
            next_state = np.reshape(next_state, [1, state_size])
            states = next_state
        # print(np.sum(QN.grad_avg/training, axis=0)/batch_size)
        QN.grad_avg = 0
        print(len(QN.critic_loss_a))
        print(e)
        print(sim_env.error/testing)
        error_avg = np.append(error_avg, np.power(10, -sim_env.error / testing))
    grad_overall = np.asarray(QN.grad_a).reshape((int(len(QN.grad_a) / QN.action_size), QN.action_size))

    dir = sim_env.channel_type
    parameters = '_DDPG_S{}_rho{}_SNR{}_PS{}_lr{}_df{}_sigOU{}_thetaOU{}'. \
        format(sim_env.S, sim_env.p, sim_env.SNR_avg[0], sim_env.pi, QN.learning_rate_actor,
               QN.gamma, OU_sigma, OU_theta)

    np.savetxt(dir + '/Error' + parameters + '.csv', np.transpose(error_avg), header='Error[sum(-log10(e))]',
               fmt='0%30.28f')
    np.savetxt(dir + '/AbsError' + parameters + '.csv', np.transpose(error), header='Absolute Error of every Cycle',
               fmt='0%30.28f')
    np.savetxt(dir + '/AvgGradient' + parameters + '.csv', grad_overall,
               header='Average_Gradient of every Batch/Training Cycle', fmt='%30.28f')
    np.savetxt(dir + '/AvgLoss_Critic' + parameters + '.csv', QN.critic_loss_a,
               header='Average_Loss of every Training from Critic Network', fmt='%30.28f')
    QN.save_weights(dir + '/', parameters)




