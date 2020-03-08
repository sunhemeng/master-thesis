import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from itertools import product

'''
Exhaustive Search of analytical solution. For different return (action, blocklength) 
and different workload assignments(equal or partitioned)
'''


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
        self.r = 320
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
        #self.action_size = len(self.action)
        self.feature = 2  # number of feature
        self.reward = 0

        #self.SNR_avg = self.P_dB - ((17 + 40 * np.log10(self.r)) + self.No_dB + 10 * np.log10(self.B))
        self.SNR_avg = snr_set
        self.SNR = []                   # initialize SNR of channel
        for i in range(self.S):
            #self.SNR = np.append(self.SNR, np.power(10, (self.P - (self.phi[i] + self.No + 10*np.log10(self.B))) / 10))
            self.SNR = np.append(self.SNR, np.power(10, self.SNR_avg / 10))
            #self.SNR = np.append(self.SNR, 3.6264 * 10**5)

    def compute_action_space(self):
        l = list(product(range(2), repeat=self.S))
        action_space = np.asarray(l)
        action_space = action_space[1:]
        return action_space

    def return_action(self, SNR):
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
                m = np.amax([t2 - ((ck + self.d) / self.f), 0])
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
                m = np.amax([t2 - ((ck + self.d) / self.f), 0])
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


# # Initialize Simulation Environment
sim_env = Optimal_Offloading(2, 1, 15)  #  number_of_server = 3, number_of_users = 1
error, m = sim_env.return_blocklength(np.power(10, sim_env.SNR_avg/10))
error_ac, ac = sim_env.return_action(np.power(10, sim_env.SNR_avg/10))
print(error)
print(m)
print(error_ac)
print(ac)
