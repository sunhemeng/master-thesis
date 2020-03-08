import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from scipy import special
from itertools import product
from keras.callbacks import History
import Sim_Optimal_Offloading_V2
from DQN import DQNAgent
history = History()

'''
Environment for loading existing weights of DQ network, and test it in environment with given parameters
'''


class Simulation:
    def __init__(self, number_of_servers=3, number_of_users=1, historic_time=4, snr_set=10, csi=0, channel=0.9):
        # define Simulation parameters
        self.S = number_of_servers  # number of servers
        self.N = number_of_users  # number of users
        self.W = historic_time
        self.features = 2  # number of feature
        self.action = self.compute_action_space()  # action set
        self.server_selection_size = len(self.action)
        self.CSI = csi        # 0 = information of previous time slot, 1 = perfect CSI
        self.p = channel      # channel correlation between time slots
        print(self.action)
        self.channel_type = self.channel_type_name()

        # Communication parameters
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
        self.pi = 320  # Packet Size in bits
        self.channel_sequence = self.create_random_channel()

        #Computation Parameters
        self.co = 24 * 10 ** 6  # total workload in cycles 24
        self.f = 3 * 10 ** 9  # computation power
        self.lam = 3 * 10 ** 6  # poisson arriving rate
        self.xi = -0.0214  # shape parameter
        self.sigma = 3.4955 * 10 ** 6  # scale parameter
        self.d = 2.0384 * 10 ** 7  # threshold
        self.T = 0.025  # delay tolerance in s 0.025
        self.eps_max = 0.001        # max error probability
        self.threshold_computation = 0.999  # computation threshold
        self.comp_correlation = 4*10**6    # computation correlation
        self.hdl_sum = 0

        # Feedback
        self.reward = 0
        self.error = 0
        #self.rnd_channel = np.random.seed(22363)
        #print(np.random.get_state())

        # Simulation
        self.h = []             # channel gain
        for i in range(self.S):
            self.h = np.append(self.h, 1)

        # self.phi = []  # propagation attenuation
        # for i in range(self.S):
        #     self.phi = np.append(self.phi, 17 + 40 * np.log10(self.r[i]))
        #
        # print(self.P_dB - (self.phi[0] + self.No_dB + 10 * np.log10(self.B)))

        ### Initialize SNR array ###
        self.SNR_s = []  # initialize SNR of channel
        self.SNR_avg = np.power(10, np.asarray(snr_set)/10)
        self.SNR = []
        for i in range(self.S):
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i])
        for i in range(self.W):
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)
        self.SNR = self.SNR.reshape((self.W, self.S))
        print(self.SNR)

        ### Initiliaze previous task sizes ###
        self.tasks_prev = []        # initiliaze previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI), self.S)
        print(self.tasks_prev)

        ### Initialize state, concatenate snr + previous tasks sizes ###
        if self.CSI == 1:
            self.state = np.concatenate((np.log10(self.SNR)/10, self.tasks_prev/(10**8)), axis=0)
        else:
            self.state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**8)), axis=0)
        # self.state = np.concatenate((np.log10(self.SNR)/10, self.tasks_prev/(10**8)), axis=0)    # initialize state

        print(self.state)

    def reset(self):
        '''
        reset reward, error, prev workload and state
        :return:
        '''
        # Feedback
        self.reward = 0
        self.error = 0

        ### Initiliaze previous task sizes ###
        self.tasks_prev = []        # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI), self.S)

        ### Initialize state, concatenate snr + previous tasks sizes ###
        if self.CSI == 1:
            self.state = np.concatenate((np.log10(self.SNR)/10, self.tasks_prev/(10**8)), axis=0)
        else:
            self.state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**8)), axis=0)

    def create_random_channel(self):
        '''
        create fixed pseudo-random channel of 1000 realizations
        :return:
        '''
        h_c = []
        for c in range(self.S):
            h_c.append([])

        for s in range(self.S):
            np.random.seed(s)
            for i in range(1000):
                h_c[s].append(complex(np.random.randn(), np.random.randn()) / np.sqrt(2))
        return h_c

    def channel_type_name(self):
        '''
        compute path for saving data
        :return:
        '''
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
        action space for server selection
        :return:
        '''
        l = list(product(range(2), repeat=self.S))
        action_space = np.asarray(l)
        action_space = action_space[1:]
        return action_space

    def total_error(self, used, n):
        '''
        calculate error based on server selection and computed blocklength
        :param used: server selection array
        :param n: blocklength
        :return:
        '''
        t1 = self.TS * n
        ck = self.co / np.count_nonzero(used) # assign equal workload
        t2 = self.T - t1
        r = self.pi / n  # coding rate


        # communication error
        shannon = np.log2(1 + self.SNR[-1])  # shannon
        channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
        Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
        comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))


        # computation error
        if self.CSI == 1:
            comp_depend = 0
        else:
            comp_depend = self.tasks_prev[-2] - self.comp_correlation
            comp_depend[comp_depend < 0] = 0

        m = t2 - (((used*ck) + self.d) + comp_depend) / self.f
        m[m < 0] = 0
        comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                        -1 / self.xi)

        total_k = used * (comm_error + comp_err - (
                        comm_error * comp_err))  # compute error of single channel + computation

        # for i in range(self.S):  # total error
        #     if total_k[i] > 0.01:
        #         total_k[i] = 1
        # print('error:', total_k)
        total = 1 - np.prod(1 - total_k)        # overall error
        # print('comm_error:', comm_error)
        # print('comp_error:', comp_err)
        # print('total_error:', total)
        return total

    # def update_channel_state(self, r):
    #     SNR = []
    #     for i in range(self.S):          # compute new SNR for every channel
    #         # h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
    #         h_bar = self.channel_sequence[i][r]
    #         hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
    #         SNR = np.append(SNR, self.SNR_avg[i] * abs(np.square(hdl)))
    #         self.h[i] = hdl
    #     self.SNR = np.append(self.SNR[1:], SNR)
    #     self.SNR = self.SNR.reshape((self.W, self.S))

    def update_channel_state(self, r):
        '''
        update channel based on fixed channel realizations
        :param r:
        :return:
        '''
        SNR = []
        for i in range(self.S):          # compute new SNR for every channel
            # h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
            h_bar = self.channel_sequence[i][r]
            # print(h_bar)
            hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
            if i >=4:
                SNR = np.append(SNR, 0.0001)
                continue
            SNR = np.append(SNR, self.SNR_avg[i] * abs(np.square(hdl)))

            self.hdl_sum += abs(hdl)
            print(self.hdl_sum / (r*self.S))
        self.SNR = np.append(self.SNR[1:], SNR)
        self.SNR = self.SNR.reshape((self.W, self.S))

    def create_state(self):
        '''
        create state based on SNR and workload array
        '''
        #state = np.concatenate((np.log10(self.SNR[:(-1+self.CSI)])/10, self.tasks_prev/(10**9)), axis=0)
        if self.CSI == 1:                   # for perfect CSI just SNR because no dependency in computation
            state = np.log10(self.SNR) / 10
        else:
            state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**8)), axis=0)
        return state

    def compute_rewards(self, error_probability):
        '''
        compute rewards
        :param error_probability:
        :return:
        '''
        e = error_probability
        if e == 0:
            self.reward = 1
            return
        if e == 1:
            self.reward = -1
            return

        self.reward = - np.log10(e)/10

    def calculate_blocklength_perfCSI(self, used):
        '''
        calculate best blocklength for given server selection using exhaustive search
        :param used: server selection
        :return: blocklength n
        '''
        T = self.T
        t1 = 1
        n_lower_bound = 100
        n_upper_bound = int(np.floor(self.T / self.TS) - 100)
        lowest = 1
        t1_lowest = 0
        ck = self.co / np.count_nonzero(used)  # assignning equal workload to all selected servers
        # time_slot_const = np.minself.comp_correlation * self.tasks_prev
        n_lowest = n_lower_bound
        for n in range(n_lower_bound, n_upper_bound, 10):     # try every blocklength in 10 steps
            t1 = n * self.TS  # compute t1 from blocklength
            t2 = T - t1  # t2
            r = self.pi / n  # coding rate

            # communication error
            shannon = np.log2(1 + self.SNR[-1])  # shannon
            channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
            Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
            comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))

            # computation error
            m = np.asarray([t2 - (((ck*used) + self.d) / self.f), [0] * self.S]).max(axis=0, initial=0)
            # m = np.asarray([m, [0] * self.S]).max(axis=0, initial=0)
            comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                        -1 / self.xi)

            total_k = used * (comm_error + comp_err - (
                        comm_error * comp_err))  # compute error of single channel + computation

            # for i in range(self.S):  # total error
            #     if total_k[i] > 0.01:
            #         total_k[i] = 1
            total = 1 - np.prod(1 - total_k)        # overall error

            if total <= lowest:         # save blocklength with lowest error
                n_lowest = t1/self.TS
                lowest = total

        self.tasks_prev = np.append(self.tasks_prev[1:], ck*used)           # update previous workload
        self.tasks_prev = self.tasks_prev.reshape((self.W-1+ self.CSI, self.S))
        return n_lowest

    def Assign_Cores(self, actions, r):
        '''
        get action from agent, compute blocklength, error, reward and updated channel for next state
        returns next state, reward and error
        '''
        ac = self.action[actions]       # take given action
        self.reward = 0                 # set reward to 0
        non_zeros = np.count_nonzero(ac)        # check if any server is choosen

        if non_zeros > 0:
            blkl = self.calculate_blocklength_perfCSI(ac)
            # print('blkl:', blkl)
            # print('SNR:', np.log10(self.SNR[-1])/100)
            total_error = self.total_error(ac, blkl)      # computes the error probability for optimal t
            self.compute_rewards(total_error)         # assigns reward

        if total_error < 10**-100:
            total_error = 10**-100
        self.error += - np.log10(total_error)
        # print('error:', total_error)
        # print('state:', self.state)
        # print('action:', ac)
        # print('SNR:', self.SNR)
        # print('reward:', self.reward)
        self.update_channel_state(r)            # update channel SNR for next state
        self.state = self.create_state()       # updates next state
        #print(self.state)
        return self.state, self.reward, total_error

loss_overall = []
error_avg = []
# for i in range(loop_end - loop_start + 1):
#     error_avg.append([])
error = []

# Initialize Simulation Environment
num_of_servers = 4  # number of servers
hist_timeslots = 4      # historic information for agent
SNR = 10.0
avg_SNR = [SNR, SNR, SNR, SNR, SNR, SNR, SNR]    # in dB
sim_env = Simulation(number_of_servers=num_of_servers, number_of_users=1, historic_time=hist_timeslots,
                    snr_set=avg_SNR, csi=0, channel=0.9)  # number_of_server = 3, number_of_users = 1, historic time slots, avgSNR, perfCSI?(yes=0, no=1), channel correlation

# arrays for different environmental parameters
T_choice = [0.005, 0.01, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.040]
co_choice = [8 * 10**6, 12 * 10**6, 16 * 10**6, 20 * 10**6, 24 * 10**6, 28 * 10**6, 32 * 10**6]
rho_choice = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
# rho_choice = [0.9]
# Q-Network
state_size = ((sim_env.features - sim_env.CSI) * (sim_env.W - 1 + sim_env.CSI) * sim_env.S) # computes state size
action_size = sim_env.server_selection_size     # set action size depending on K
nl = 1     # number of hidden layers
sl_f = 1    # size of layers factor to state size
expl_decay = 0.999     # exploration reduction
QN = DQNAgent(state_size, action_size, discount_factor=0.0, learning_rate=0.001,
                  expl_decay=expl_decay, nhl=nl, sl_f=sl_f)
batch_size = 16

# Simulation Runtime
testing = 1000
testing_comps = 1

for r in range(4,5):
    # sim_env.T = T_choice[r]
    # load Q network
    print(r)
    parameters = 'old/DQN__S{}_rho{}_SNR{}_PS{}_W4_lr0.0001_df0.0_sl24_nhl1_ef0.9'.format(r, 0.9, SNR,
                                                                                      sim_env.pi)
    QN.load(sim_env.channel_type + '/' + parameters)
    sim_env = Simulation(number_of_servers=num_of_servers, number_of_users=1, historic_time=hist_timeslots,
                         snr_set=avg_SNR, csi=0, channel=0.9)
    sim_env.reset()
    states = sim_env.state      # get first state
    for u in range(testing):
        states = np.reshape(states, [1, state_size])        # reshape state to vector for network
        action = QN.act_test(states)            # get action from DQN agent
        # print('SNR:', sim_env.SNR[-1])
        # print('action:', sim_env.action[action])
        next_state, rewards, overall_err = sim_env.Assign_Cores(action, u)      # get next state, reward and error
        error = np.append(error, overall_err)
        next_state = np.reshape(next_state, [1, state_size])        # reshape next state
        states = next_state         # state = next state
    print(r)
    print(np.power(10, -sim_env.error/testing))
    error_avg = np.append(error_avg, np.power(10, -sim_env.error/testing))

# Save Error and Losses in CSV file, Save weights of networks ####
parameters = 'DQN_S{}_rho{}_SNR{}_PS{}_OverMaxBlkL'.format(sim_env.S, sim_env.p, sim_env.SNR_avg[0], sim_env.pi)
np.savetxt(sim_env.channel_type + '/Error' + parameters + '.csv', np.transpose(error_avg),
               header='Error[sum(-log10(e))]', fmt='0%30.28f')



