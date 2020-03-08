import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from scipy import special
from itertools import product
from keras.callbacks import History
import Sim_Optimal_Offloading_V2
from DQN_double_dueling import Double_DDQNAgent
from DQN import DQNAgent
history = History()

'''
Learn an agent in environment. Training and testing phase.
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
        self.pi = 320  # Packet Size in bits

        # Computation Parameters
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
        # self.rnd_channel = np.random.seed(22363)
        # print(np.random.get_state())

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
        self.SNR_avg = np.power(10, [np.asarray(snr_set)/10]*self.S)
        self.SNR = []
        for i in range(self.S):
            self.SNR_s = np.append(self.SNR_s, self.SNR_avg[i])
        for i in range(self.W):
            self.SNR = np.concatenate((self.SNR, self.SNR_s), axis=0)          # ratio
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
        #self.state = np.concatenate((np.log10(self.SNR[:(-1+self.CSI)])/10, self.tasks_prev/(10**9)), axis=0)    # initialize state
        if self.CSI == 1:
            self.state = np.concatenate((np.log10(self.SNR)/10, self.tasks_prev/(10**8)), axis=0)       # CSI in dB/100
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

        # Initialize previous task sizes ###
        self.tasks_prev = []        # previous task size
        self.tasks_prev_s = []
        for i in range(self.S):
            self.tasks_prev_s = np.append(self.tasks_prev_s, 0)
        for i in range(self.W - 1 + self.CSI):
            self.tasks_prev = np.concatenate((self.tasks_prev, self.tasks_prev_s), axis=0)
        self.tasks_prev = self.tasks_prev.reshape((self.W - 1 + self.CSI), self.S)

        # Initialize state, concatenate snr + previous tasks sizes ###
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
        ck = self.co / np.count_nonzero(used)
        t2 = self.T - t1
        r = self.pi / n  # coding rate

        shannon = np.log2(1 + self.SNR[-1])  # shannon
        channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
        Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
        comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))

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
        total = 1 - np.prod(1 - total_k)
        # print('comm_error:', comm_error)
        # print('comp_error:', comp_err)
        # print('total_error:', total)
        return total

    def update_channel_state(self):
        '''
        update channel based on fixed channel realizations
        :param r:
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
        create state based on SNR and workload array
        '''
        if self.CSI == 1:
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
        ck = self.co / np.count_nonzero(used)
        # time_slot_const = np.minself.comp_correlation * self.tasks_prev

        for n in range(n_lower_bound, n_upper_bound, 10):
            t1 = n * self.TS  # compute t1 from blocklength
            t2 = T - t1  # t2
            r = self.pi / n  # coding rate
            shannon = np.log2(1 + self.SNR[-1])  # shannon
            channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
            Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
            comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))

            comp_depend = 0

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

            if total <= lowest:
                n_lowest = t1/self.TS
                lowest = total

        self.tasks_prev = np.append(self.tasks_prev[1:], ck*used)
        self.tasks_prev = self.tasks_prev.reshape((self.W-1+ self.CSI, self.S))
        return n_lowest

    def Assign_Cores(self, actions):
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
        self.update_channel_state()            # update channel SNR for next state
        self.state = self.create_state()       # updates next state
        #print(self.state)
        return self.state, self.reward, total_error


hist_timeslots = 1      # historic information for agent
SNR = 10
avg_SNR = [0.8*SNR, 1*SNR, 1.2*SNR]    # in dB

loop_start = 2
loop_end = 10
loss_overall = []
error_avg = []
for i in range(loop_end - loop_start + 1):
    error_avg.append([])
error = []

for r in range(loop_start, loop_end + 1):
    num_of_servers = r  # number of servers
    # Initialize Simulation Environment
    sim_env = Simulation(number_of_servers=num_of_servers, number_of_users=1, historic_time=hist_timeslots,
                         snr_set=SNR, csi=1, channel=0.9)  # number_of_server = 3, number_of_users = 1, historic time slots, avgSNR, perfCSI?(yes=0, no=1), channel correlation

    episodes = 200
    training = 50
    testing = 1000
    testing_comps = 10

    # Q-Network
    state_size = ((sim_env.features-sim_env.CSI) * (sim_env.W - 1 + sim_env.CSI) * sim_env.S)
    action_size = sim_env.server_selection_size
    nl = 2     # number of hidden layers
    sl_f = 2    # size of layers factor to state size
    expl_decay = 0.999     # exploration reduction
    QN = Double_DDQNAgent(state_size, action_size, discount_factor=0.0, learning_rate=0.01,
                  expl_decay=expl_decay, nhl=nl, sl_f=sl_f)

    batch_size = 16
    error = []
    loss_overall = []
    # Q network learning and testing
    #optimal_off = Sim_Optimal_Offloading_V2.Optimal_Offloading(r, 1)
    states = sim_env.state
    for e in range(episodes):
        sim_env.reset()
        for k in range(training):
            states = np.reshape(states, [1, state_size])
            action = QN.act(states)
            # print(sim_env.SNR)
            next_state, rewards, overall_err = sim_env.Assign_Cores(action)
            next_state = np.reshape(next_state, [1, state_size])
            QN.remember(states, action, rewards, next_state)
            states = next_state
            if len(QN.memory) > batch_size:
                QN.replay(batch_size)

        loss_overall = np.append(loss_overall, QN.loss_avg/training)
        QN.loss_avg = 0

        sim_env.reset()
        for u in range(testing):
            states = np.reshape(states, [1, state_size])
            action = QN.act_test(states)
            # print('SNR:', sim_env.SNR[-1])
            # print('action:', sim_env.action[action])
            next_state, rewards, overall_err = sim_env.Assign_Cores(action)
            error = np.append(error, overall_err)
            next_state = np.reshape(next_state, [1, state_size])
            states = next_state
        print(e)
        print(sim_env.error/testing)
        error_avg[r-loop_start] = np.append(error_avg[r-loop_start], np.power(10, -sim_env.error/testing))

### Save Error and Losses in CSV file, Save weights of networks ####
    parameters = 'DDQN_S{}_rho{}_SNR{}_PS{}_adjHP'.format(sim_env.S, sim_env.p, SNR, sim_env.pi)
    np.savetxt(sim_env.channel_type + '/Error' + parameters + '.csv',
               np.transpose(error_avg[r-loop_start]), header='Error[sum(-log10(e))]', fmt='0%30.28f')
    np.savetxt(sim_env.channel_type + '/Abs_Error' + parameters + '.csv',
               np.transpose(error), header='Error', fmt='0%13.11f')
    np.savetxt(sim_env.channel_type + '/Avg_Loss' + parameters + '.csv',
               np.transpose(loss_overall), header='Error[sum(-log10(e))]', fmt='0%30.28f')
    np.savetxt(sim_env.channel_type + '/Abs_Loss' + parameters + '.csv',
               np.transpose(QN.loss), header='Error', fmt='0%13.11f')
    QN.save(sim_env.channel_type + '/' + parameters)

### Test Using random number of available Servers ###
sim_env.reset()
for u in range(testing_comps):
    action = QN.random_act()
    next_state, rewards, overall_err = sim_env.Assign_Cores(action)
avg_error_random = np.power(10, -sim_env.error / testing_comps)
print(avg_error_random)

## Test with optimal current decision ###
optimal_current = Sim_Optimal_Offloading_V2.Optimal_Offloading(sim_env.S, 1, sim_env.SNR_avg[0])
sim_env.reset()
for u in range(testing_comps):
    error_opt, action = optimal_current.return_action(sim_env.SNR[-1])
    next_state, rewards, overall_err = sim_env.Assign_Cores(action)
avg_error_current_optimal = np.power(10, -sim_env.error/testing_comps)
print(avg_error_current_optimal)


##Test using all available Servers
# sim_env.error = 0
# sim_env.loss = []
# for u in range(testing_comps):
#     states = np.reshape(states, [1, state_size])
#     action = QN.all_act()
#     next_state, rewards, overall_err = sim_env.Assign_Cores(action)
#     #print(sim_env.action[action])
#     if overall_err <= sim_env.eps_max:
#         sim_env.optimum_count = sim_env.optimum_count + 1
#     sim_env.loss = np.append(sim_env.loss, (np.log10(overall_err) - np.log10(sim_env.eps_max)) ** 2)
#     sim_env.error += overall_err
#     next_state = np.reshape(next_state, [1, state_size])
#     states = next_state
#error_avg[r - loop_start + 1] = np.append(error_avg[r - loop_start + 1], sim_env.error / testing)
# error_avg[r - loop_start + 2] = [sim_env.error/testing] * episodes
# error_optimum_avg_all = np.sum(sim_env.loss) / testing_comps
# print(error_avg[r - loop_start +2])


for i in range(loop_end - loop_start + 1):
     plt.figure(i)
     #plt.plot(error_optimum[0], '-b.',error_optimum[1], '-r.', error_optimum[2], '-g.', markersize=1)
     plt.plot(error, 'b.', markersize=1.5)
     plt.xlabel('Cycles')
     plt.ylabel('eps_RL')
     plt.title('Absolute Error probability' .format(loop_start + i))
     #plt.legend(['K=2', 'K=3', 'K=4'])
     plt.axis([0, episodes*testing, 0, 10**2])
     #ax1.set_yscale('symlog', nonposy='clip', linthreshy=0.01)
     plt.yscale('symlog', nonposy='clip', linthreshy=10**-8)
     plt.grid()

plt.figure(loop_end - loop_start + 2)
#plt.plot(error_optimum[0], '-b.',error_optimum[1], '-r.', error_optimum[2], '-g.', markersize=1)
plt.plot(loss_overall, '-b.', markersize=1.5)
plt.xlabel('Episodes')
plt.ylabel('Loss')
plt.title('Average Loss for 250 training cycles' )
#plt.legend(['K=2', 'K=3', 'K=4'])
#plt.axis([0, episodes*testing, 0, 10**2])
#ax1.set_yscale('symlog', nonposy='clip', linthreshy=0.01)
#plt.yscale('symlog', nonposy='clip', linthreshy=10**-8)
plt.grid()

plt.figure(loop_end - loop_start + 4)
#plt.plot(error_optimum[0], '-b.',error_optimum[1], '-r.', error_optimum[2], '-g.', markersize=1)
plt.plot(QN.loss, 'b.', markersize=0.75)
plt.xlabel('Cycles')
plt.ylabel('Loss')
plt.title('Absolute Loss')
#plt.legend(['K=2', 'K=3', 'K=4'])
#plt.axis([0, episodes*testing, 0, 10**2])
#ax1.set_yscale('symlog', nonposy='clip', linthreshy=0.01)
#plt.yscale('symlog', nonposy='clip', linthreshy=10**-8)
plt.grid()

#error_avg_diff = abs(np.log10(np.asarray(error_avg)) - np.log10(avg_error_random))
#print(error_avg)
plt.figure(loop_end - loop_start + 3)
plt.plot(error_avg[0], '-g.', [avg_error_current_optimal]*episodes, '-b.', [avg_error_random]*episodes, '-r.', markersize=1)
#plt.plot([avg_error_random]*episodes, '-r.', error_avg[0], '-g.', error_avg[1], '-b.' [avg_error_current_optimal]*episodes, '-y.', markersize=1)
plt.axis([0, episodes, 0, 1.1])
plt.xlabel('Episodes')
plt.ylabel('Error Probability')
plt.title('Average Error Probability for Non-Perfect CSI(avg {}dB)'
          ' with {} Servers, p={}, comp={}'.format(sim_env.SNR_avg[0], sim_env.S, sim_env.p, sim_env.comp_correlation))
plt.legend([ 'RL with {} Servers'.format(sim_env.S)])
plt.yscale('symlog', nonposy='clip', linthreshy=10**-9)
plt.grid()
plt.show()