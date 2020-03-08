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

# lsfont = {'fontname':'lmodern'}
plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams["font.family"] = ["lmodern"]
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"legend.fontsize": 8})


# Color initilization for plots
blue = (0, 0.328125, 0.62109375)
black = (0, 0, 0)
petrol = (0, 0.37890625, 0.39453125)
petrol2 = (0.17578125, 0.49609375, 0.51171875)
tuerkis = (0, 0.59375, 0.62890625)
orange = (0.9609375, 0.65625, 0, 1)
darkred = (0.7, 0.01, 0.1)
red = (0.796875, 0.02734375, 0.1171875, 1)
hellred = (0.84375, 0.359375, 0.253906425, 1)
hellred2 = (0.8984375, 0.5859375, 0.47265625, 1)
winered = (0.62890625, 0.0625, 0.20703125, 1)
hellwinered = (0.7109375, 0.3203125, 0.3359375, 1)
hellwinered2 = (0.80078125, 0.54296875, 0.52734375, 1)
violett = (0.37890625, 0.12890625, 0.34375, 1)
hellviolett = (0.51171875, 0.3046875, 0.45703125, 1)
hellviolett2 = (0.65625, 0.51953125, 0.6171875, 1)
violett2 = (0.6953125, 0.2265625, 0.9296875, 1)
lila = (0.4765625, 0.43359375, 0.671875, 1)
helllila = (0.60546875, 0.56640625, 0.75390625, 1)
helllila2 = (0.734375, 0.70703125, 0.83984375, 1)
hellmagenta = (0.94140625, 0.6171875, 0.69140625, 1)
green2 = (0.33984375, 0.66796875, 0.15234375, 1)
# color_array = [red, hellred, hellred2]  # lr
# color_array = [darkred, 'red', hellwinered, hellwinered2, ]  # df
# color_array = [violett, violett2, hellviolett, hellviolett2, ]  # sigma
color_array = [hellmagenta, lila, helllila2]  # nhl

'''
Loads existing weights of actor and critic network. Tests these networks in an environment. 
Networks architecture (number of layer, neurons, activation function, have to be known)
'''

class Simulation:
    def __init__(self, number_of_servers, number_of_users, historic_time, snr_set):
        # define Simulation parameters
        self.S = number_of_servers  # number of servers
        self.N = number_of_users  # number of users
        self.W = historic_time
        self.features = 2  # number of feature
        self.action = self.compute_action_space()  # action set
        self.server_selection_size = len(self.action)
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
        self.pi = 320 # bits
        self.channel_sequence = self.create_random_channel()

        self.co = 24 * 10 ** 6  # total workload in cycles 24
        self.f = 3 * 10 ** 9  # computation power
        self.lam = 3 * 10 ** 6  # poisson arriving rate
        self.xi = -0.0214  # shape parameter
        self.sigma = 3.4955 * 10 ** 6  # scale parameter
        self.d = 2.0384 * 10 ** 7  # threshold
        self.T = 0.025  # delay tolerance in s 0.025
        self.eps_max = 0.001
        self.threshold_computation = 0.999  # computation threshold
        self.p = 0.9      # channel correlation between time slots
        self.comp_correlation = 0.00001     # computation correlation
        self.comm_error = 0
        self.comp_error = 0
        self.snr = snr_set
        # Feedback
        self.reward = 0
        self.error = 0
        self.rnd_channel = np.random.seed(2236391566)
        # print(np.random.get_state())

        # Simulation
        self.h = []             # channel gain
        for i in range(self.S):
            self.h = np.append(self.h, complex(np.random.randn(), np.random.randn()) / np.sqrt(2))
            # self.h = np.append(self.h, 1)

        self.SNR_s = []  # initialize SNR of channel
        self.SNR_avg = [np.power(10, self.snr/10)]*self.S       # set average SNR of every channel
        for i in range(self.S):
            # self.SNR_avg = np.append(self.SNR_avg, np.power(10, (self.P_dB - (self.phi[i] + self.No_dB + 10*np.log10(self.B))) / 10))
            #self.SNR_s = np.append(self.SNR_s, np.power(10, (self.P_dB - (self.phi[i] + self.No_dB + 10*np.log10(self.B))) / 10))
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i] * abs(np.square(self.h[i])))
            # self.SNR_s = np.append(self.SNR_s, self.SNR_avg[0])

        self.SNR = []
        for i in range(self.W):
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)           # set full SNR array

        self.SNR = self.SNR.reshape((self.W, self.S))
        print(self.SNR)
        self.tasks_prev = []        # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W-1):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W-1, self.S))           # set full workload array
        print(self.tasks_prev)
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

        self.tasks_prev = []  # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1, self.S))

        self.state = np.concatenate((np.log10(self.SNR[:-1]) / 10, self.tasks_prev / (10 ** 9)),
                                    axis=0)  # initialize state

    def create_random_channel(self):
        '''
        create channel sequence h^
        :return: returns h^ for every server with length 1000
        '''
        h_c = []
        for c in range(self.S):
            h_c.append([])
        for s in range(self.S):
            np.random.seed(s)
            for i in range(10000):
                h_c[s].append(complex(np.random.randn(), np.random.randn()) / np.sqrt(2))
        return h_c

    def compute_action_space(self):
        '''
        computes action space dependent on partitioned workload. Has to set sizes of partioned workloads.
        :return: action space matrix regarding workload.
        '''
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
        r = self.pi / n  # coding rate
        shannon = np.log2(1 + self.SNR[-1])  # shannon
        channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
        Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
        comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))

        # if self.CSI == 1:
        #     comp_depend = 0
        # else:
        comp_depend = self.tasks_prev[-1] - self.comp_correlation
        comp_depend[comp_depend < 0] = 0

        used = np.clip(ck, 0, 1)
        m = t2 - (((ck + self.d) + comp_depend) / self.f)
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
        # print('total_k:', total_k)
        # print('SNR:', self.SNR[-1])
        # print('comm:', comm_error)
        # print('comp:', comp_err)
        # print('total:', total)
        return total

    def update_channel_state(self, r):
        '''
        Updates channel based on pseudo-random channel sequence
        :param count: Cycle counter
        :return: update of SNR array
        '''
        SNR = []
        for i in range(self.S):          # compute new SNR for every channel
            # h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
            h_bar = self.channel_sequence[i][r]
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
        self.reward = - np.log10(e)/100

    def Assign_Cores(self, actions,r):
        '''
        :param actions: workload index
        :param count: Cycle counter for pseudo-random channel sequence
        :param train: Training or testing (1 for training)=> which channel update should be taken?
        :return: next state, reward, error
        '''
        action_blkl = int(round((actions[0]) * (self.T/self.TS))) # take given action

        action_selection = self.action[int(round(actions[1]*(len(self.action) -1)))]
        self.reward = 0                 # set reward to 0
        non_zeros = np.count_nonzero(action_selection)        # check if any server is choosen

        if non_zeros > 0:
            #blkl = self.calculate_blocklength(ac)
            total_error = self.total_error(action_blkl, action_selection)      # computes the error probability for optimal t
            self.compute_rewards(total_error)         # assigns reward
            if total_error < 10**-25:
                total_error = 10**-25
            self.error += - np.log10(total_error)

        self.update_channel_state(r)            # update channel SNR for next state
        self.state = self.create_state()       # updates next state
        #print(self.state)
        return self.state, self.reward, total_error

loop_start = 2  # start number of servers
loop_end = 2    # end number of servers
loss_overall = []
error_avg = []
num_servers = 3
for i in range(loop_end - loop_start + 1):
    error_avg.append([])

# combination array of different environment setups
comb = [0, 20*10**6, 40*10**6, 60*10**6, 80*10**6, 100*10**6, 120*10**6]  # workload
# comb = [0.015, 0.02, 0.025, 0.03, 0.035]      # Time delay

error = []
for i in range(len(comb)):
    error.append([])

print(comb)
for r in range(loop_start, loop_end + 1):
    # Initialize Simulation Environment
    sim_env = Simulation(r, 1, 4, 10)  # number_of_server = 3, number_of_users = 1, historic time slots, avgSNR
    testing = 1000      # number of testing cycles


    # Q-Network
    state_size = (sim_env.features * (sim_env.W-1) * sim_env.S)  # compute state size
    action_size = 2         # action size (blocklength and partitioned workload index)
    batch_size = 512        # batch size

    # Initialize DDPG agent and networks
    QN = DDPGAgent(state_size=state_size, action_size=2, gamma=0.0, learning_rate_actor=0.00005, learning_rate_critic=0.0001, tau=0.001, batch_size=batch_size, action_max=[1, 1])
    # Initialize Noise process
    Noise = OUNoise(action_space=action_size,  mu=np.asarray([0.0, 0.0]), theta=np.asarray([0.5, 0.5]), max_sigma=np.asarray([0.8, 0.9]), min_sigma=np.asarray([0.1, 0.05]), action_max=[1, 1], action_min=[0.001, 0])

    # load networks weights for actor and critic
    QN.load_weights('FadingChannel_OutdatedCSI/MA-raw-results/Weights_DDPG_S{}_rho0.9_SNR10.0_PS320_lr5e-05_df0.0_W4_sigOU0.8_thetaOU0.5_Critic_LR_weights_actor_h5'.format(2),
                    'FadingChannel_OutdatedCSI/MA-raw-results/Weights_DDPG_S{}_rho0.9_SNR10.0_PS320_lr5e-05_df0.0_W4_sigOU0.8_thetaOU0.5_Critic_LR_weights_critic_h5'.format(2))


    # sim_env.T = 0.025
    # sim_env.p = 0.9
    index = 0
    for v in comb:
        # QN = DDPGAgent(state_size=state_size, action_size=2, gamma=0.0, learning_rate_actor=0.00005,
        #                learning_rate_critic=0.0001, tau=0.001, batch_size=batch_size, action_max=[1, 1])
        # Noise = OUNoise(action_space=action_size, mu=np.asarray([0.0, 0.0]), theta=np.asarray([0.5, 0.5]),
        #                 max_sigma=np.asarray([0.8, 0.9]), min_sigma=np.asarray([0.1, 0.05]), action_max=[1, 1],
        #                 action_min=[0.001, 0])
        # QN.load_weights(
        #     'FadingChannel_OutdatedCSI/critic/Weights_DDPG_S{}_rho{}_SNR10.0_PS320_lr5e-05_df0.0_W4_sigOU0.8_thetaOU0.5_Critic_LR_weights_actor_h5'.format(
        #         r, v),
        #     'FadingChannel_OutdatedCSI/critic/Weights_DDPG_S{}_rho{}_SNR10.0_PS320_lr5e-05_df0.0_W4_sigOU0.8_thetaOU0.5_Critic_LR_weights_critic_h5'.format(
        #         r, v))
        sim_env = Simulation(r, 1, 4, 10)
        print(v)
        sim_env.co = v
        #sim_env.T = v * 10**-3
        # Q network learning and testing
        #optimal_off = Sim_Optimal_Offloading_V2.Optimal_Offloading(r, 1)
        states = sim_env.state
        sim_env.reset()
        for u in range(testing):
            states = np.reshape(states, [1, state_size])
            action = np.clip(QN.policy_action(states), [0.001, 0], [1, 1])
            next_state, rewards, overall_err = sim_env.Assign_Cores(action, u)
            error[index] = np.append(error[index], overall_err)
            next_state = np.reshape(next_state, [1, state_size])
            states = next_state
        QN.grad_avg = 0
        #print(e)
        print(np.power(10, -sim_env.error/testing))
        error_avg[r-loop_start] = np.append(error_avg[r-loop_start], np.power(10, -sim_env.error/testing))
        index += 1

parameters = 'S{}_rho{}_SNR{}_PS{}_OverCO'.format(sim_env.S, sim_env.p, sim_env.SNR_avg[0], sim_env.pi)
np.savetxt('Error' + parameters + '.csv', np.transpose(error_avg), header='Error[sum(-log10(e))] for c0=[0,20,40,60,80,100,120]', fmt='0%30.28f')
np.savetxt('Abs_Error' + parameters + '.csv', np.transpose(error), header='AbsError[sum(-log10(e))] for c0=[0,20,40,60,80,100,120]', fmt='0%30.28f')
# np.savetxt('Error' + parameters + '.csv', np.transpose(error_avg), header='Error[sum(-log10(e))] for c0=[0.015, 0.02, 0.025, 0.03, 0.035]', fmt='0%30.28f')
# np.savetxt('Abs_Error' + parameters + '.csv', np.transpose(error), header='AbsError[sum(-log10(e))] for c0=[0.015, 0.02, 0.025, 0.03, 0.035]', fmt='0%30.28f')
print(error_avg)
print(error_avg[0])


# # Large 1-Plot
# parameters = 'S{}_rho{}_SNR{}_PS{}'.format(sim_env.S, sim_env.p, sim_env.SNR_avg[0], sim_env.pi)
# np.savetxt('Error' + parameters + '.csv', np.transpose(error_avg), header='Error[sum(-log10(e))]', fmt='0%11.9f')
# QN.save_weights(parameters)
# sum_sign_err = '$' + r'\v' + 'arepsilon_{O,log}$'
# sum_sign_loss = '$L_{log}$'
# sum_grad = '$' + r'\f' + 'rac{1}{N}\sum_{i}' + r'\n' + 'abla_{a} Q(s, a| ' + r'\t' + 'heta^{Q})$'
# var = r'$\v' + 'arepsilon$'
# rho = r'$\r' + 'ho$'
# fig = plt.figure(figsize=(5, 3.75), dpi=1000)
# ax = fig.add_axes([0.05, 0.05, 0.85, 0.85])
# x_data = list(range(2, 8))
# ax.grid(True, linewidth=0.1, which='both')
# # print(np.asarray(comb)*10**-3)
# # ax.plot(x_data, opt_K3, color=(0.9609375, 0.65625, 0, 0.9), linestyle=(0, (1, 1)), linewidth=1.25, marker='v', markersize=2, rasterized=True) #fading pCSI
# ax.plot(np.asarray(comb), error_avg[0], color=red, linestyle='--',
#         linewidth=2, marker='o', markersize=3, markerfacecolor=blue,  markeredgecolor=blue, rasterized=True)
# # ax.plot(np.asarray(comb)*10**-3, error_avg[1], color=hellred, linestyle='--',
# #         linewidth=2, marker='o', markersize=3, markerfacecolor=hellmagenta, markeredgecolor=hellmagenta, rasterized=True)
# # ax.plot(np.asarray(comb)*10**-3, error_avg[2], color=hellred2, linestyle='dotted',
# #         linewidth=2, marker='X', markersize=3, markerfacecolor=green2, markeredgecolor=green2, rasterized=True)
# ax.vlines(24*10**6, ymin=0, ymax=1, linestyle='dashed', linewidth=1)
# # ax.plot(comb, opt_K3[:-1], color=orange, linestyle=(0, (1, 1)), linewidth=2, marker='x',
# #         markersize=2, markerfacecolor=violett,  markeredgecolor=lila, rasterized=True)                         # static
# # ax.plot(x_data, error_last, color=petrol, linestyle='dashed', linewidth=1, marker='>', markersize=1.25, rasterized=True)
# # ax.plot(x_data, error_best, color=petrol, linestyle='dotted', linewidth=1, marker='<', markersize=1.25, rasterized=True)
#
# ax.spines['left'].set_position(('axes', 0))
# ax.spines['bottom'].set_position(('axes', 0))
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
#
# leg = fig.legend(['$K=3$', 'Trained'], facecolor='white', framealpha=1, fancybox=False, edgecolor='black',
#                  loc='upper center', bbox_to_anchor=(0.535, 1), ncol=2)
#
# leg.get_frame().set_linewidth(0.1)
# ax.set_xlim(0, 120*10**6)
# ax.set_ylim(0, 1)
# ax.tick_params(which='minor', width=0.2)
# ax.set_xlabel('$c_o$ in [Cycles]')
# # ax.set_xlabel('Channel correlation factor' + rho)
# # ax.set_ylabel('$' + r'\v' + 'arepsilon_{O}$')
# ax.set_ylabel(sum_sign_err)
# ax.set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6, 8], linthreshy=10**-8)
#
# fig.tight_layout()
# plt.savefig('06-DDPG_diffT_' + '.pdf', bbox_inches='tight', dpi=1000, frameon=False)
# # plt.show()





