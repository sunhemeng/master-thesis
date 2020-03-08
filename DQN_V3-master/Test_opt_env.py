import numpy as np
from scipy import special
from itertools import product
from SIm_Optimal_Offloading_V2_plot import Optimal_Offloading

class Simulation:
    def __init__(self, number_of_servers=3, number_of_users=1, historic_time=4, snr_set=10, csi=0, channel=0.9, exp=1):
        # define Simulation parameters
        self.S = number_of_servers  # number of servers
        self.N = number_of_users  # number of users
        self.W = historic_time
        self.features = 2  # number of feature
        self.action = self.compute_action_space_work()  # action set
        self.server_selection_size = len(self.action)
        self.CSI = csi        # 0 = information of previous time slot, 1 = perfect CSI
        self.p = channel      # channel correlation between time slots
        # print(self.action)
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
            # self.h = np.append(self.h, complex(np.random.randn(), np.random.randn()) / np.sqrt(2))

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
        # print(self.SNR)

        self.channel_sequence = self.create_random_channel_V2(exp)
        print(len(self.channel_sequence[0]))

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
        l = list(product(range(2), repeat=self.S))
        action_space = np.asarray(l)
        action_space = action_space[1:]
        return action_space

    def create_random_channel(self, exp):
        h_c = []
        for c in range(self.S):
            h_c.append([])

        for s in range(self.S):
            np.random.seed(s)
            for i in range(1000):
                h_c[s].append(complex(np.random.randn(), np.random.randn()) / np.sqrt(2))
        return h_c

    def create_random_channel_V2(self, exp):
        h_c = []
        SNR = []

        for c in range(self.S):
            h_c.append([])
            SNR.append([])

        for s in range(self.S):
            np.random.seed(s)
            for i in range(1000*exp):
                h_c[s].append(complex(np.random.randn(), np.random.randn()) / np.sqrt(2))
        for i in range(self.S):          # compute new SNR for every channel
            for k in range(1000*exp):
                # print(k)
                h_bar = h_c[i][k]
                # h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
                hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
                if np.mod(k, exp) == 0:
                    SNR[i] = np.append(SNR[i], self.SNR_avg[i] * abs(np.square(hdl)))
                self.h[i] = hdl
                # self.SNR = self.SNR.reshape((self.W, self.S))))
        print('len', len(SNR[0]))
        return SNR

    def compute_action_space_work(self):
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

    def total_error(self, ck_in, n):
        # print(n)
        t1 = self.TS * n
        ck = self.co / np.count_nonzero(ck_in)
        used = np.clip(ck, 0, 1)
        # print(ck)
        t2 = self.T - t1
        r = self.pi / n  # coding rate
        shannon = np.log2(1 + self.SNR[-1])  # shannon
        channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
        Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
        comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
        # print('comm-error:', comm_error)
        if self.CSI == 1:
            comp_depend = 0
        else:
            comp_depend = self.tasks_prev[-1] - self.comp_correlation
            comp_depend[comp_depend < 0] = 0

        m = t2 - ((((used*ck) + self.d) + comp_depend) / self.f)
        m[m < 0] = 0
        comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                     -1 / self.xi)
        # print('comp-error:', comp_err)
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

    def update_channel_state(self, r):
        SNR = []
        for i in range(self.S):          # compute new SNR for every channel
            h_bar = self.channel_sequence[i][r]
            # h_bar = complex(np.random.randn(), np.random.randn()) / np.sqrt(2)
            hdl = self.p * self.h[i] + (np.sqrt(1 - np.square(self.p)) * h_bar)
            SNR = np.append(SNR, self.SNR_avg[i] * abs(np.square(hdl)))
            self.h[i] = hdl
        self.SNR = np.append(self.SNR[1:], SNR)
        self.SNR = self.SNR.reshape((self.W, self.S))

    def update_channel_state_sequence(self, r):
        SNR_k = []
        for i in range(self.S):
            SNR_k.append(self.channel_sequence[i][r])
        SNR_k = np.asarray(SNR_k).reshape((1, self.S))
        self.SNR = np.append(self.SNR[1:], SNR_k)
        self.SNR = self.SNR.reshape((self.W, self.S))

    def create_state(self):
        #  Creates state dependent on buffer
        #  :return:
        # state = np.concatenate((np.log10(self.SNR[:(-1+self.CSI)])/10, self.tasks_prev/(10**9)), axis=0)
        if self.CSI == 1:
            state = np.concatenate((np.log10(self.SNR)/10, self.tasks_prev/(10**8)), axis=0)
        else:
            state = np.concatenate((np.log10(self.SNR[:-1])/10, self.tasks_prev/(10**8)), axis=0)
        return state

    def compute_rewards(self, error_probability):
        #  reward function: max(-log(error_probability))
        #  :update reward variable
        e = error_probability
        if e == 0:
            self.reward = 1
            return
        if e == 1:
            self.reward = -1
            return
        self.reward = - np.log10(e)/10

    def calculate_blocklength_perfCSI(self, used):
        T = self.T
        t1 = 1
        n_lower_bound = 100
        n_upper_bound = int(np.floor(self.T / self.TS) - 100)
        lowest = 1
        t1_lowest = 0
        ck = self.co / np.count_nonzero(used)
        # time_slot_const = np.minself.comp_correlation * self.tasks_prev
        # n_lowest = n_lower_bound
        for n in range(n_lower_bound, n_upper_bound, 10):
            t1 = n * self.TS  # compute t1 from blocklength
            t2 = T - t1  # t2
            r = self.pi / n  # coding rate
            shannon = np.log2(1 + self.SNR[-1])  # shannon
            channel_disp = 1 - (1 / (1 + self.SNR[-1]) ** 2)  # channel dispersion
            Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
            comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
            if self.CSI == 1:
                comp_depend = 0
            else:
                comp_depend = self.tasks_prev[-1] - self.comp_correlation
                comp_depend[comp_depend < 0] = 0

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
        return n_lowest, lowest

    def Assign_Cores(self, actions, ep, n):
        #  Assigns cores to tasks from buffers.
        #  :return: next state, reward, total error
        ac = self.action[actions]       # take given action
        self.reward = 0                 # set reward to 0
        non_zeros = np.count_nonzero(ac)        # check if any server is choosen
        # print(ac)
        if non_zeros > 0:
            # blkl, total_error = self.calculate_blocklength_perfCSI(ac)
            # print(blkl)
            # print(total_error)
            # print('blkl:', blkl)
            # print('SNR:', np.log10(self.SNR[-1])/100)
            total_error = self.total_error(ac, n)      # computes the error probability for optimal t
            self.compute_rewards(total_error)         # assigns reward
        if total_error < 10**-100:
            total_error = 10**-100
        self.error += - np.log10(total_error)
        # print('error:', total_error)
        # print('state:', self.state)
        # print('action:', ac)
        # print('SNR:', self.SNR)
        # print('reward:', self.reward)
        self.update_channel_state_sequence(ep)            # update channel SNR for next state
        self.state = self.create_state()       # updates next state
        #print(self.state)
        return self.state, self.reward, total_error

SNR = 10
# avg_SNR = [0.8*SNR, 1*SNR, 1.2*SNR]    # in dB
loop_start = 2
loop_end = 11

error_avg = []
error = []
# rho_a = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
exp = [1, 5, 9, 13, 17, 21]

for r in exp:
    num_of_servers = 3  # number of servers
    # Initialize Simulation Environment
    sim_env = Simulation(number_of_servers=num_of_servers, number_of_users=1, historic_time=2,
                         snr_set=SNR, csi=0, channel=0.9, exp=r)  # number_of_server = 3, number_of_users = 1, historic time slots, avgSNR, perfCSI?(yes=0, no=1), channel correlation
    opt_env = Optimal_Offloading(num_of_servers, 1, SNR)
    testing_comps = 1000

    error = []
    states = sim_env.state
    for e in range(testing_comps):
        error, n, action = opt_env.run_opt_sim_work(sim_env.SNR[-2])
        # error, action = opt_env.return_action(sim_env.SNR[-2])
        # print(action)
        # print('n',n)
        # print('opt::', error)
        # print(sim_env.SNR)
        # print(action)
        next_state, rewards, overall_err = sim_env.Assign_Cores(action, e, n)
        #print('result:', overall_err)
        error = np.append(error, overall_err)
        #print(e)
    #print(sim_env.SNR)
    error_avg.append(np.power(10, -(sim_env.error/testing_comps)))
    print('Error:', error_avg)

# Save Error and Losses in CSV file, Save weights of networks ####
# parameters = '_Opt_Equal_S{}_rho{}_SNR{}_PS{}_pCSI'.format(sim_env.S, sim_env.p, SNR, sim_env.pi)
parameters = '_Opt_overRho_OCSI_V2'
np.savetxt('05-Error' + parameters + '.csv', np.transpose(error_avg), header='Error[sum(-log10(e))]', fmt='0%30.28f')


# np.savetxt('Abs_Error' + parameters + '.csv',
#               np.transpose(error), header='Error', fmt='0%13.11f')