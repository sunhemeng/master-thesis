import matplotlib.pyplot as plt
import random
import numpy as np
from collections import deque
from scipy import special
from scipy.stats import rayleigh
from itertools import product
#import Sim_Optimal_Offloading_V2
from DDPG import DDPGAgent
import h5py
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
        self.CSI = csi        # 0 = information of previous time slot, 1 = perfect CSI
        self.p = channel      # channel correlation between time slots

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
        self.comp_correlation = 4*10**6    # computation correlation
        # evaluation variables
        self.comm_error = 0
        self.comp_error = 0
        # Feedback
        self.reward = 0
        self.error = 0
        #self.rnd_channel = np.random.seed(22363)
        #print(np.random.get_state())

        self.channel_type = self.channel_type_name()
        print(self.channel_type)

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
        self.SNR_avg = [np.power(10, snr_set/10)]*self.S
        self.SNR = []
        for i in range(self.S):
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i])
        for i in range(self.W):
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)
        self.SNR = self.SNR.reshape((self.W, self.S))
        print(self.SNR)

        ### Initiliaze previous task sizes ###
        self.tasks_prev = []        # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI), self.S)
        print(self.tasks_prev)

        ### Initialize state, concatenate snr + previous tasks sizes ###
        self.state = np.concatenate((np.log10(self.SNR[:(-1+self.CSI)])/10, self.tasks_prev/(10**9)), axis=0)    # initialize state
        print(self.state)

    def reset(self):
        # Feedback
        self.reward = 0
        self.error = 0

        self.tasks_prev = []  # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1, self.S))
        print(self.tasks_prev)

        ### Initialize state, concatenate snr + previous tasks sizes ###
        self.state = np.concatenate((np.log10(self.SNR[:-1]) / 10, self.tasks_prev / (10 ** 9)),
                                    axis=0)  # initialize state

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
        # l = list(product(range(2), repeat=3))
        l = list(product([0, 4, 8, 16, 24], repeat=self.S))
        # act_all = np.asarray(l)
        act_all = l[1:]
        act_space = []
        for i in range(len(act_all)):
            if np.sum(act_all[i]) == 24:
                act_space.append(act_all[i])
        #print(np.asarray(act_space))
        return np.asarray(act_space)*10**6

    def total_error(self, n, c):
        t1 = self.TS * n
        ck = c
        t2 = self.T - t1
        r = self.pi / n  # coding rate
        shannon = np.log2(1 + self.SNR[-1])  # shannon
        channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
        Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
        comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))

        comp_depend = self.tasks_prev[-2] - self.comp_correlation
        comp_depend[comp_depend < 0] = 0
        used = np.clip(ck, 0, 1)
        # print('comp_depend:', comp_depend)
        m = t2 - (((used * ck) + self.d) + comp_depend) / self.f
        m[m < 0] = 0
        comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                        -1 / self.xi)


        total_k = used * (comm_error + comp_err - (
                        comm_error * comp_err))  # compute error of single channel + computation

        # for i in range(self.S):  # total error
        #     if total_k[i] > 0.01:
        #         total_k[i] = 1

        total = 1 - np.prod(1 - total_k)
        self.tasks_prev = np.append(self.tasks_prev[1:], ck)
        self.tasks_prev = self.tasks_prev.reshape((self.W-1, self.S))
        self.comp_error = comp_err
        self.comm_error = comm_error
        # print('SNR:', self.SNR)
        # print('comm:', comm_error)
        # print('comp:', comp_err)
        # print('total:', total)
        return total

    def update_channel_state(self):
        SNR = []
        for i in range(self.S):          # compute new SNR for every channel
            h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
            hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
            SNR = np.append(SNR, self.SNR_avg[i] * abs(np.square(hdl)))
            self.h[i] = hdl
        self.SNR = np.append(self.SNR[1:], SNR)
        self.SNR = self.SNR.reshape((self.W, self.S))

    def create_state(self):
        #  Creates state dependent on buffer
        #  :return:
        #print(np.log10(self.SNR[-1])/10)
        state = np.concatenate((np.log10(self.SNR[:(-1+self.CSI)])/10, self.tasks_prev/(10**9)), axis=0)
        #print(state)
        return state

    def compute_rewards(self, error_probability):
        #  reward function: max(-log(error_probability))
        #  :update reward variable

        e = error_probability

        if e == 0:
            self.reward = 1
            return
        #
        # if e == 1:
        #     self.reward = -1
        #     return

        self.reward = - np.log10(e)/100


    def Assign_Cores(self, actions):
        #  Assigns cores to tasks from buffers.
        #  :return: next state, reward, total error
        #print('snr:', self.SNR)
        action_blkl = int(round((actions[0]) * (self.T/self.TS)))   # take given action
        # action_blkl = 0.125 * (self.T/self.TS)
        workload_rel = actions[1:]
        if np.count_nonzero(workload_rel) == 0:
            workload_rel = [1]*self.S
        # workload_rel = [1] * self.S
        sum_work = np.sum(workload_rel)
        workload = (workload_rel/sum_work) * self.co
        # action_selection = self.action[int(round(actions[1]*(len(self.action) - 1)))]
        # print('server_selection:', action_selection)
        # print('state:', self.state)
        # print('SNR', self.SNR)
        self.reward = 0                 # set reward to 0
        # non_zeros = np.count_nonzero(actions)        # check if any server is choosen
        # print('workload')
        # print(workload)

        total_error = self.total_error(action_blkl, workload)      # computes the error probability for optimal t
        self.compute_rewards(total_error)         # assigns reward
        if total_error < 10**-100:
            total_error = 10**-100
        self.error += - np.log10(total_error)

        # print('error')
        # print(total_error)
        # print(self.reward)
        # print('loop')
        # print(total_error)
        # print(action_blkl)
        self.update_channel_state()            # update channel SNR for next state
        self.state = self.create_state()       # updates next state
        #print(self.state)
        return self.state, self.reward, total_error

    def plot_data(self, x=None, y=None, title=None, legend=None, xl=None, yl=None, scale=None):
        min = np.min(y)
        max = np.max(y)
        print(np.ndim(y))
        for i in range(np.ndim(y)):
            plt.plot(y[i])
        if scale == 'log':
            plt.yscale('symlog', nonposy='clip', linthreshy=10 ** -9)
        #plt.axis([0, len(x), min, max])
        plt.xlabel(xl)
        plt.ylabel(yl)
        plt.legend(legend)
        plt.title(title)
        plt.grid()
        plt.show()


loss_overall = []
error = []
error_avg = []
T_choice = [0.005, 0.01, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03, 0.0325, 0.035, 0.0375, 0.040]
num_of_servers = 3         # number of servers
hist_timeslots = 4      # historic information for agent
avg_SNR = 10    # in dB


for i in range(len(T_choice)):
    error.append([])

# Initialize Simulation Environment
sim_env = Simulation(number_of_servers=num_of_servers, number_of_users=1, historic_time=hist_timeslots,
                         snr_set=avg_SNR, csi=0, channel=0.9)  # number_of_server = 3, number_of_users = 1, historic time slots, avgSNR, perfCSI?, channel correlation

# Q-Network
state_size = (sim_env.features * (sim_env.W - 1) * sim_env.S)
action_size = sim_env.S + 1
batch_size = 1000
print('se')
print(state_size)

QN = DDPGAgent(state_size=state_size, action_size=2, gamma=0.5, learning_rate_actor=0.0001,
               learning_rate_critic=0.00005, tau=0.0001, batch_size=batch_size, action_max=[1] * (sim_env.S + 1))
model = QN.load_weights(sim_env.channel_type + '/MA-raw-results/Weights_DDPG_S{}_rho{}_SNR{}_PS{}_lr5e-05_df0.0_W4_sigOU0.8_thetaOU0.5_Critic_LR_weights_actor_h5'.format(sim_env.S, sim_env.p,
                                                                                            sim_env.SNR_avg[0],
                                                                                            sim_env.pi),
                sim_env.channel_type + '/MA-raw-results/Weights_DDPG_S{}_rho{}_SNR{}_PS{}_lr5e-05_df0.0_W4_sigOU0.8_thetaOU0.5_Critic_LR_weights_critic_h5'.format(sim_env.S, sim_env.p,
                                                                                             sim_env.SNR_avg[0],
                                                                                             sim_env.pi))
# f = h5py.File(sim_env.channel_type + '/MA-raw-results/Weights_S{}_rho{}_SNR{}_PS{}_LR_weights_actor_h5'.format(sim_env.S, sim_env.p,
#                                                                                             sim_env.SNR_avg[0],
#                                                                                             sim_env.pi), 'r')


# print('model', list(f.keys()))
# # will get a list of layer names which you can use as index
# d = f['model_weights']
# print(d.shape)
# <HDF5 dataset "kernel:0": shape (128, 1), type "<f4">
# d.shape == (128, 1)
# d[0] == array([-0.14390108], dtype=float32)

# QN.load_model(sim_env.channel_type + '/WeightsS{}_rho{}_SNR{}_PS{}_V2_LR_actor_h5'.format(sim_env.S, sim_env.p,
#                                                                                             sim_env.SNR_avg[0],
#                                                                                             sim_env.pi),
#                 sim_env.channel_type + '/WeightsS{}_rho{}_SNR{}_PS{}_V2_LR_critic_h5'.format(sim_env.S, sim_env.p,
#                                                                                              sim_env.SNR_avg[0],
#                                                                                              sim_env.pi))
# print(QN.actor.model)

# for r in range(len(T_choice)):
#     sim_env.reset()
#     sim_env.T = T_choice[r]
#     testing = 1
#     # Q network learning and testing
#     #optimal_off = Sim_Optimal_Offloading_V2.Optimal_Offloading(r, 1)
#     states = sim_env.state
#
#     for u in range(testing):
#         states = np.reshape(states, [1, state_size])
#         action = np.clip(QN.policy_action(states), [0.001, 0, 0, 0], [1]*(sim_env.S+1))
#         # print('##################test#######################')
#         # print('action:', action)
#         next_state, rewards, overall_err = sim_env.Assign_Cores(action)
#         # print('reward:', rewards)
#         # print(overall_err)
#         # print(rewards)
#         error[r] = np.append(error[r], overall_err)
#         next_state = np.reshape(next_state, [1, state_size])
#         states = next_state
#     print(r)
#     print(np.power(10, -sim_env.error/testing))
#     error_avg = np.append(error_avg, np.power(10, -sim_env.error/testing))
#
# parameters = 'S{}_rho{}_SNR{}_PS{}'.format(sim_env.S, sim_env.p, sim_env.SNR_avg[0], sim_env.pi)
# np.savetxt(sim_env.channel_type + '/Error' + parameters + '_OverMaxBlkL.csv', np.transpose(error_avg), header='Error[sum(-log10(e))]', fmt='0%30.28f')
# # np.savetxt('Abs_Error' + parameters + '_6.csv', np.transpose(error), header='Error', fmt='0%13.11f')
#
# #error_avg_diff = abs(np.log10(np.asarray(error_avg)) - np.log10(avg_error_random))
# #print(QN.loss)
# # print(avg_error_current_optimal)
# plt.figure(1)
# #plt.plot(error_avg[0], '-b.', error_avg[1], '-r.', error_avg[2], '-g.', error_avg[3], '-y.', error_avg[4], '-k.', markersize=1)
# plt.plot(np.asarray(T_choice)/sim_env.TS, error_avg, '-g*', markersize=2)
# #plt.plot(error_avg[0], '-b.', markersize=1)
# plt.axis([min(T_choice)*0.75, max(T_choice)/sim_env.TS, 0, 1])
# plt.xlabel('Max_blocklength')
# plt.ylabel('-log10(eps)')
# plt.title('Error Probability for static channel (avg {}dB) with {} Servers'.format(sim_env.SNR_avg[0], sim_env.S))
# plt.legend(['eps_RL with {} Servers'.format(sim_env.S)])
# #plt.yscale('symlog', nonposy='clip', linthreshy=10**-20)
# plt.grid()
# plt.show()