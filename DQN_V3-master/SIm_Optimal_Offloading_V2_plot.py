import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from itertools import product

'''
Exhaustive Search of analytical solution. For different return (action, blocklength) 
and different workload assignments(equal or partitioned)
'''

plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams["font.family"] = ["lmodern"]
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"legend.fontsize": 6})
# plt.rcParams["figure.figsize"] = [2.835, 2.126]
# plt.rcParams["figure.dpi"] = 1000

blue = (0, 0.328125, 0.62109375)
black = (0, 0, 0)
petrol = (0, 0.37890625, 0.39453125)
orange = (0.9609375, 0.65625, 0)
red = (0.62890625, 0.0625, 0.20703125)
lila = (0.4765625, 0.43359375, 0.671875)
yellow = {1, 0.92578125, 0}
green = {0.339884375, 0.66796875, 0.15234375}

class Optimal_Offloading:
    def __init__(self, number_of_servers, number_of_users, snr_set):
        # define constants
        self.S = number_of_servers  # number of servers
        self.N = number_of_users  #  number of users
        self.area_width = 50
        self.area_length = 50
        self.poisson_variable = np.random.poisson(0.5*self.area_width, 3)
        self.A = self.area_length * self.area_width  # area in square meter
        self.B = 5 * 10**6    # bandwidth
        self.F = 2.4 * 10**9  # carrier frequency
        self.r = 340
        self.No_dBm = -174  # noise power level in dbm/Hz
        self.No_dB = self.No_dBm - 30
        self.TS = 0.025 * 10**-3  # symbol time
        self.P_dBm = 20  # in dbm
        self.P_dB = self.P_dBm - 30
        self.pi = 320  # bits
        self.co = 24 * 10**6  # total workload in cycles
        self.f = 3 * 10**9  # computation power
        self.lam = 3 * 10**6  # poisson arriving rate
        self.xi = -0.0214  # shape parameter
        self.sigma = 3.4955 * 10**6  # scale parameter
        self.d = 2.0384 * 10**7  # threshold
        self.T = 0.025  # delay tolerance in s
        self.eps_max = 0.001
        self.threshold_computation = 0.999  # computation threshold
        self.action = self.compute_action_space()  # action set
        self.action_work = self.compute_action_space_work()
        #self.action_size = len(self.action)
        self.feature = 2  # number of feature
        self.reward = 0

        self.tot_err_a = []
        self.comm_err_a = []
        self.comp_err_a = []
        self.tot_err_k_a = []

        #self.SNR_avg = self.P_dB - ((17 + 40 * np.log10(self.r)) + self.No_dB + 10 * np.log10(self.B))
        self.SNR_avg = snr_set
        self.SNR = []                   # initialize SNR of channel
        for i in range(self.S):
            #self.SNR = np.append(self.SNR, np.power(10, (self.P - (self.phi[i] + self.No + 10*np.log10(self.B))) / 10))
            self.SNR = np.append(self.SNR, np.power(10, self.SNR_avg / 10))
            #self.SNR = np.append(self.SNR, 3.6264 * 10**5)

    def compute_action_space(self):
        '''
        server selection
        :return: server selection array
        '''
        l = list(product(range(2), repeat=self.S))
        action_space = np.asarray(l)
        action_space = action_space[1:]
        return action_space

    def compute_action_space_work(self):
        '''
        partitioned workload
        :return: workload array
        '''
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

    def return_action(self, SNR):
        '''
        action space is server selection array A
        :param SNR:
        :return: error, a
        '''
        T = self.T
        lowest_total = 1
        best_action_i = 0
        for index in range(len(self.action)):
            n_lower_bound = 100
            n_upper_bound = int(np.floor(self.T / self.TS) - 100)
            lowest = 1
            action = self.action[index]
            ck = self.co / np.count_nonzero(action)
            err_values_single = []
            t1_values_single = []
            for n in range(n_lower_bound, n_upper_bound, 10):
                t1 = n * self.TS
                t2 = T - t1
                r = self.pi / n
                Q_arg = np.sqrt(n / (1 - (1 / (1 + SNR)) ** 2)) * (np.log2(1 + SNR) - r) * np.log(2)
                comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))

                comp_depend = 0
                m = t2 - ((((action * ck) + self.d) + comp_depend) / self.f)
                m[m < 0] = 0
                comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                            -1 / self.xi)


                # if t2 < (self.d / self.f):
                #     comp_err = 1

                total_k = action * (comm_error + comp_err - (comm_error * comp_err))

                # for i in range(self.S):
                #     if total_k[i] > 0.01:
                #         total_k[i] = 1

                total = 1 - np.prod(1 - total_k)
                err_values_single = np.append(err_values_single, total_k[0])
                t1_values_single = np.append(t1_values_single, t1)
                if total < lowest:
                    t1_lowest = t1
                    lowest = total
            if lowest < lowest_total:
                lowest_total = lowest
                best_action_i = index

        return lowest_total, best_action_i

    def return_blocklength(self, SNR):
        '''
        action space is server selection array A
        :param SNR:
        :return: error, n
        '''
        T = self.T
        lowest_total = 1
        for index in range(len(self.action)):
            n_lower_bound = 100
            n_upper_bound = int(np.floor(self.T / self.TS) - 100)
            lowest = 1
            best_action = []
            action = self.action[index]
            ck = self.co / np.count_nonzero(action)
            err_values_single = []
            t1_values_single = []
            t1_best = n_lower_bound* self.TS
            for n in range(n_lower_bound, n_upper_bound, 10):
                t1 = n * self.TS
                t2 = T - t1
                r = self.pi / n
                #shannon = np.log2(1 + self.SNR)
                #channel_disp = 1 - (1 / (1 + self.SNR) ** 2)
                Q_arg = np.sqrt(n / (1 - (1 / (1 + SNR)) ** 2)) * (np.log2(1 + SNR) - r) * np.log(2)
                comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
                comp_depend = 0
                # else:
                #     comp_depend = self.tasks_prev[-2] - self.comp_correlation
                #     comp_depend[comp_depend < 0] = 0
                # used = np.clip(ck, 0, 1)
                m = t2 - ((((action * ck) + self.d) + comp_depend) / self.f)
                m[m < 0] = 0
                comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                            -1 / self.xi)
                # if t2 < (self.d / self.f):
                #     comp_err = 1
                total_k = action * (comm_error + comp_err - (comm_error * comp_err))
                # for i in range(self.S):
                #     if total_k[i] > 0.01:
                #         total_k[i] = 1
                total = 1 - np.prod(1 - total_k)
                err_values_single = np.append(err_values_single, total_k[0])
                t1_values_single = np.append(t1_values_single, t1)
                if total < lowest:
                    t1_lowest = t1
                    lowest = total

            if lowest < lowest_total:
                lowest_total = lowest
                t1_best = t1_lowest

        return lowest_total, t1_best/self.TS

    def run_opt_sim(self, SNR):
        '''
        action space is server selection array A
        :param SNR:
        :return: error, n
        '''
        snr = np.asarray([SNR] * self.S)
        T = self.T
        lowest_total = 1
        action_space = self.action
        # action_space = [[0, 1, 1]]
        for index in range(len(action_space)):
            n_lower_bound = 100
            n_upper_bound = int(np.floor(self.T / self.TS) - 100)
            lowest = 1
            best_action = []
            action = np.asarray(action_space[index])
            ck = self.co / np.count_nonzero(action)
            err_values_single = []
            t1_values_single = []
            t1_best = n_lower_bound * self.TS
            for n in range(n_lower_bound, n_upper_bound, 10):
                t1 = n * self.TS
                t2 = T - t1
                r = self.pi / n

                shannon = np.log2(1 + snr)  # shannon
                channel_disp = 1 - (1 / (1 + snr) ** 2)  # channel dispersion
                Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
                comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
                self.comm_err_a.append(comm_error[0])
                # if self.CSI == 1:
                comp_depend = 0
                # else:
                #     comp_depend = self.tasks_prev[-2] - self.comp_correlation
                #     comp_depend[comp_depend < 0] = 0
                # used = np.clip(ck, 0, 1)
                m = t2 - ((((action * ck) + self.d) + comp_depend) / self.f)
                m[m < 0] = 0
                comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                        -1 / self.xi)
                self.comp_err_a.append(comp_err[0])
                # if t2 < (self.d / self.f):
                #     comp_err = 1
                total_k = action * (comm_error + comp_err - (comm_error * comp_err))
                # for i in range(self.S):
                #     if total_k[i] > 0.01:
                #         total_k[i] = 1
                total = 1 - np.prod(1 - total_k)

                self.tot_err_k_a = np.append(self.tot_err_a, total_k[0])
                if total < lowest:
                    t1_lowest = t1
                    lowest = total
                self.tot_err_a.append(total)
            if lowest < lowest_total:
                lowest_total = lowest
                t1_best = t1_lowest

        return lowest_total, t1_best / self.TS

    def run_opt_sim_work(self, SNR):
        '''
        action space is workload array
        :param SNR:
        :return: error, n, c
        '''
        snr = np.asarray(SNR)
        T = self.T
        lowest_total = 1
        action_space = self.action_work
        # action_space = np.asarray([[0, 8*10**6, 16*10**6]])
        for index in range(len(action_space)):
            n_lower_bound = 100
            n_upper_bound = int(np.floor(self.T / self.TS) - 100)
            lowest = 1
            best_action = []
            ck = np.asarray(action_space[index])
            err_values_single = []
            t1_values_single = []
            t1_best = n_lower_bound * self.TS
            for n in range(n_lower_bound, n_upper_bound, 10):
                t1 = n * self.TS
                t2 = T - t1
                r = self.pi / n

                shannon = np.log2(1 + snr)  # shannon
                channel_disp = 1 - (1 / (1 + snr) ** 2)  # channel dispersion
                Q_arg = np.sqrt(n / channel_disp) * (shannon - r) * np.log(2)
                comm_error = 0.5 * special.erfc(Q_arg / np.sqrt(2))
                self.comm_err_a.append(comm_error)
                # if self.CSI == 1:
                comp_depend = 0
                # else:
                #     comp_depend = self.tasks_prev[-2] - self.comp_correlation
                #     comp_depend[comp_depend < 0] = 0
                used = np.clip(ck, 0, 1)
                # print(ck)
                m = t2 - ((ck + self.d) + comp_depend) / self.f
                m[m < 0] = 0
                comp_err = (1 - self.threshold_computation) * (1 + ((self.xi * m) / (self.sigma / self.f))) ** (
                        -1 / self.xi)
                self.comp_err_a.append(comp_err[0])
                # if t2 < (self.d / self.f):
                #     comp_err = 1
                total_k = used * (comm_error + comp_err - (comm_error * comp_err))
                # for i in range(self.S):
                #     if total_k[i] > 0.01:
                #         total_k[i] = 1
                total = 1 - np.prod(1 - total_k)

                self.tot_err_k_a = np.append(self.tot_err_a, total_k[0])
                if total < lowest:
                    t1_lowest = t1
                    lowest = total
                    ck_b = ck
                self.tot_err_a.append(total)
            if lowest < lowest_total:
                lowest_total = lowest
                t1_best = t1_lowest
                ck_b_tot = ck_b

        return lowest_total, t1_best / self.TS, ck_b_tot

# sim_env = Optimal_Offloading(3, 1, 10)  #  number_of_server = 3, number_of_users = 1
# error, n_best, ck = sim_env.run_opt_sim_work(np.power(10, 20/10))
# print('ErrorK=3:', error)
#
#
# plot_control = 0
# snr_a = np.asarray([10])
# k_data = range(2, 11)
# # snr_range = range(10, 11, 2)
# opt_err_a = []
# per_S_total_err = []
# for k in snr_a:
#     for i in k_data:
#         sim_env = Optimal_Offloading(i, 1, 10)  #  number_of_server = 3, number_of_users = 1 initialize environment
#         error, n_best = sim_env.run_opt_sim(np.power(10, k/10))       # get server selection a
#         opt_err_a = np.append(opt_err_a, error)           # save error
#         print(k)
#         print(i)
#         print(error)
#         print(n_best)
#
#     per_S_total_err = np.append(per_S_total_err, opt_err_a)
#     opt_err_a = []
#
# per_S_total_err = per_S_total_err.reshape((len(snr_a), len(k_data)))
# print(np.transpose(per_S_total_err))
# np.savetxt('Optimal_OverK_DiffSNR.csv', np.transpose(per_S_total_err),
#            delimiter=', ', header='SNR (0;510;15;20dB) over K(2-10)', fmt=['0%15.14f', '0%15.14f', '0%15.14f', '0%15.14f', '0%15.14f'])
# print(per_S_total_err)
# # Plot over Server Number
# alpha_c = np.asarray(list(range(1,  11, 2)))/10
# print(alpha_c)
# fig = plt.figure(figsize=(2.85, 2.3), dpi=2000)
# ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
# ax.plot(k_data, per_S_total_err[0], color=(0.9609375, 0.65625, 0, 0.2), linestyle='-', linewidth=1)
# ax.plot(k_data, per_S_total_err[1], color=(0.9609375, 0.65625, 0, 0.4), linestyle='-', linewidth=1)
# ax.plot(k_data, per_S_total_err[2], color=(0.9609375, 0.65625, 0, 0.6), linestyle='-', linewidth=1)
# ax.plot(k_data, per_S_total_err[3], color=(0.9609375, 0.65625, 0, 0.8), linestyle='-', linewidth=1)
# ax.plot(k_data, per_S_total_err[4], color=(0.9609375, 0.65625, 0, 1), linestyle='-', linewidth=1)
# var = r'$\v' + 'arepsilon'
# leg = ax.legend(['SNR=0dB', 'SNR=5dB', 'SNR=10dB', 'SNR=15dB', 'SNR=20dB'], facecolor='white', framealpha=1, fancybox=False, edgecolor='black')
# leg.get_frame().set_linewidth(0.1)
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
# ax.set_xlim(min(k_data), max(k_data))
# ax.set_ylim(0, 1)
# ax.tick_params(axis='both', direction='in')
# ax.tick_params(which='minor', width=0.2)
# # ax.yaxis.yticks(y_ticks)
# # ax.xaxis.set_ticks([-5, -3, -1, 1, 3, 5])
# # ax.yaxis.set_ticks([10**-10, 10**-9, 10**-8, 10**-4, 10**-3, 1])
# ax.set_xlabel('Number of Servers $K$')
# ax.set_ylabel(var + '_O$')
# plt.grid(linewidth=0.1)
# fig.tight_layout()
# ax.set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6, 8], linthreshy=10 ** -10)
# plt.savefig('Opt_Err_ForKServers.pdf', bbox_inches='tight', dpi=1000, frameon=False)
# plt.show
# plt.close(fig)
#
# if plot_control == 1:
#     sim_env = Optimal_Offloading(3, 1, 10)  #  number_of_server = 3, number_of_users = 1
#     error, n_best = sim_env.run_opt_sim_work(np.power(10, snr_a/10))
#     # error, n_best, = sim_env.run_opt_sim(np.power(10, snr_a / 10))
#
#     # Plot Communication, Computation, Channel and Total Error
#     print(len(sim_env.comp_err_a))
#     print(error)
#     print(n_best)
#     y_ticks = [10**-10, 10**-9, 10**-8, 10**-4, 10**-3, 1]
#     x_data = range(1, 1000, 10)
#     # fig = plt.figure(figsize=(6, 4.5), dpi=1000)
#     fig = plt.figure(figsize=(2.85, 2.3), dpi=2000)
#     ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
#     ax.plot(x_data, sim_env.comm_err_a[0], color=(0.62890625, 0.0625, 0.20703125, 0.2), linestyle='--', linewidth=1)
#     ax.plot(x_data, sim_env.comm_err_a[1], color=(0.62890625, 0.0625, 0.20703125, 0.6), linestyle='--', linewidth=1)
#     ax.plot(x_data, sim_env.comm_err_a[2], color=(0.62890625, 0.0625, 0.20703125, 1), linestyle='--', linewidth=1)
#     # ax.plot(x_data, sim_env.tot_err_k_a, color='r', linestyle='-.', linewidth=1)
#     ax.plot(x_data, sim_env.tot_err_a, color=orange, linestyle='-', linewidth=1)
#     lines = ax.get_lines()
#     var = r'$\v' + 'arepsilon'
#     leg = ax.legend(labels=[var + '_1$', var + '_2$', var + '_O$'], facecolor='white', framealpha=1, fancybox=False, edgecolor='black')
#     leg.get_frame().set_linewidth(0.1)
#     ax.spines['right'].set_visible(False)
#     ax.spines['top'].set_visible(False)
#     ax.set_xlim(0, (sim_env.T/sim_env.TS))
#     ax.set_ylim(0, 1)
#     ax.tick_params(axis='both', direction='in')
#     # ax.yaxis.yticks(y_ticks)
#     # ax.xaxis.set_ticks([-5, -3, -1, 1, 3, 5])
#     # ax.yaxis.set_ticks([10**-10, 10**-9, 10**-8, 10**-4, 10**-3, 1])
#     ax.set_xlabel('blocklength $n$')
#     ax.set_ylabel(var + '$')
#     plt.grid(linewidth=0.1)
#     fig.tight_layout()
#     plt.yscale('symlog', nonposy='clip', linthreshy=10**-15)
#     plt.savefig('Optimal_BlkL.pdf', bbox_inches='tight', dpi=1000, frameon=False)
#     # plt.show()