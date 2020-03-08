import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from matplotlib.ticker import MultipleLocator
from tikzplotlib import save as tikz_save
# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']

'''
Plot Error, Loss 
'''


# lsfont = {'fontname':'lmodern'}
plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams["font.family"] = ["lmodern"]
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"legend.fontsize": 8})


black = (0, 0, 0)
orange = (0.9609375, 0.65625, 0)
rot = (0.62890625, 0.0625, 0.20703125)
lila = (0.4765625, 0.43359375, 0.671875)

darkred = (0.7, 0.01, 0.1)
red = (0.796875, 0.02734375, 0.1171875, 1)
petrol = (0, 0.37890625, 0.39453125, 1)
petrol1 = (0.17578125, 0.49609375, 0.51171875, 1)
petrol2 = (0.48828125, 0.640625, 0.65234375, 1)
petrol3 = (0, 0.45890625, 0.45453125, 1)
tuerkis = (0, 0.59375, 0.62890625, 1)
tuerkis2 = (0.53515625, 0.796875, 0.80859375, 1)
green2 = (0.33984375, 0.66796875, 0.15234375, 1)
maigreen = (0.73828125, 0.80078125, 0, 1)
maigreen2 = (0.8125, 0.84765625, 0.359375, 1)
blue = (0, 0.328125, 0.62109375, 1)
hellblue = (0.28, 0.60609375, 0.75484375, 1)
mediumblue = (0.175, 0.49, 0.68)
hellgreen = (0.55078125, 0.75, 0.375, 1)
darkblue = (0, 0.25, 0.5, 1)
hellgreen2 = (0.55078125, 0.75, 0.375, 1)
hellblue2 = (0.28, 0.60609375, 0.75484375, 1)
hellmaigreen = (0.8125, 0.84765625, 0.359375, 1)
violett2 = (0.6953125, 0.2265625, 0.9296875, 1)
# color_array = ['green', green2, hellgreen, maigreen, maigreen2]   # lr
# color_array = [darkblue, blue, mediumblue, hellblue]   # df
# color_array = [petrol3, petrol2, tuerkis, tuerkis2]  # eps
color_array = [hellgreen2, hellblue2, hellmaigreen] # nhl

#  static alpha = 0.6, fading-perfectCSI: alpha= 0.8, fading-Outdated alpha = 1
# Hyperparameters DQN: türkis DDPG: rot
sp = list(product([0, 1], [0, 1]))
print(sp)
class plot_desc(object):
    def __init__(self, title, x_label, y_label, x_axis, y_axis, legend, marker):
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.x_axis = x_axis
        self.y_axis = y_axis
        self.legend = legend
        self.marker = marker

def plot_data(data, x_axis, y_axis, x_label, y_label, title, legend, log, id, dir, marker):
    print(x_axis)
    x_data = list(range(x_axis[0], x_axis[1]))
    fig_no = id
    plt.figure(fig_no, figsize=(2.85, 2.3), dpi=1000)
    plt.plot(x_data, data, marker, markersize=1, rasterized=True)
    # plt.plot([avg_error_random]*episodes, '-r.', error_avg[0], '-g.', error_avg[1], '-b.' [avg_error_current_optimal]*episodes, '-y.', markersize=1)
    plt.axis([x_axis[0], x_axis[1], y_axis[0], y_axis[1]])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.title(title)
    plt.legend(legend)
    if log == 1:
        plt.yscale('symlog', nonposy='clip', linthreshy=10 ** -9)
    plt.grid()
    # tikz_save(dir + title + '.tex')
    plt.savefig(dir + title + '.jpg')
    # plt.show()
    plt.close()

def plot_data2(data, x_axis, y_axis, x_label, y_label, title, legend, log, id, dir, marker, color, point, thresh):
    fig = plt.figure(figsize=(2.95, 2.214), dpi=1000)
    ax = fig.add_axes([0.26, 0.26, 0.7, 0.7])
    x_data = list(range(0, len(data)))
    ax.grid(True, linewidth=0.1, which='both')
    if point == 1:
        ax.plot(x_data, data, color=color, marker=marker, linewidth=1, rasterized=True)
    else:
        ax.plot(x_data, data, color=color, linestyle=marker, linewidth=1, rasterized=True)

    ax.spines['left'].set_position(('axes', 0))
    ax.spines['bottom'].set_position(('axes', 0))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if legend != 0:
        leg = fig.legend(legend, facecolor='white', framealpha=1, fancybox=False, edgecolor='black', loc='upper center')
        leg.get_frame().set_linewidth(0.1)
    ax.set_xlim(x_axis[0], x_axis[1])
    ax.set_ylim(y_axis[0], y_axis[1])
    ax.tick_params(which='minor', width=0.2)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if log == 1:
        ax.set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6,  8], linthreshy=thresh)

    fig.tight_layout()
    plt.savefig(dir + title + '.pdf', bbox_inches='tight', dpi=1000, frameon=False)
    plt.show()
    #   End Plot

def plot_data_4box(data1, data2, data3, data4, x_axis, y_axis, x_label, y_label, title, legend, log, id, dir, marker):
    ID = id
    x_data = list(range(0, len(data1)))
    x_data2 = list(range(0, len(data2)))
    x_data4 = list(range(0, len(data4)))
    fig, ax = plt.subplots(2, 2, figsize=(6, 4.5), dpi=4000)

    # ax[0][0] = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax[0, 0].plot(x_data, data1, '-', color=orange, linewidth=1, rasterized=True)
    ax[0, 1].plot(x_data2, data2, '.', color=(0.996, 0.9256, 0),  markersize=1, rasterized=True)
    ax[1, 0].plot(x_data, data3, '-', color=petrol, linewidth=1, rasterized=True)
    ax[1, 1].plot(x_data4, data4, '.', color=(0.996, 0.9256, 0), markersize=0.5, rasterized=True)
    sum_sign_err = r'$\f' + 'rac{\sum' + r'\v' + 'arepsilon_O}' + '{M}$'
    sum_sign = r'$\f' + 'rac{\sum L}{M}$'
    var = r'$\v' + 'arepsilon'
    leg = fig.legend(['DQN'], facecolor='white', framealpha=1, fancybox=False, edgecolor='black', loc='upper center')
    leg.get_frame().set_linewidth(0.1)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    for s in sp:
        ax[s].spines['left'].set_position(('axes', 0))
        ax[s].spines['bottom'].set_position(('axes', 0))
        ax[s].spines['right'].set_visible(False)
        ax[s].spines['top'].set_visible(False)
        ax[s].grid(True, linewidth=0.1)
    ax[0, 0].set_xlim = (min(x_data), max(x_data))
    ax[0, 1].set_xlim = (min(x_data), max(x_data2))
    ax[1, 0].set_xlim = (min(x_data), max(x_data))
    ax[1, 1].set_xlim = (min(x_data), max(x_data4))
    ax[0, 0].set_ylim(0, 1)
    ax[0, 1].set_ylim(0, 1)
    ax[1, 0].set_ylim(0, 1)
    ax[1, 1].set_ylim(0, 1)
    # ax[1, 0].set_ylim(0, 1)
    # ax[1, 1].set_ylim(0, 1)
    # plt.tick_params(axis='both', direction='in')
    # ax.yaxis.yticks(y_ticks)
    # ax.xaxis.set_ticks([-5, -3, -1, 1, 3, 5])
    # ax.yaxis.set_ticks([10**-10, 10**-9, 10**-8, 10**-4, 10**-3, 1])
    ax[0, 0].set_xlabel('Episodes')
    ax[0, 1].set_xlabel('Cycles')
    ax[1, 0].set_xlabel('Episodes')
    ax[1, 1].set_xlabel('Cycles')
    ax[0, 0].set_ylabel(sum_sign_err)
    ax[0, 1].set_ylabel(var + '_O$')
    ax[1, 0].set_ylabel(sum_sign)
    ax[1, 1].set_ylabel('$L$')

    ax[0, 0].set_yscale('symlog', nonposy='clip', linthreshy=10 ** -10)
    ax[0, 1].set_yscale('symlog', nonposy='clip', linthreshy=10 ** -10)
    ax[1, 0].set_yscale('symlog', nonposy='clip', linthreshy=10 ** -10)
    ax[1, 1].set_yscale('symlog', nonposy='clip', linthreshy=10 ** -10)
    fig.tight_layout()
    plt.savefig(dir + title + '.pdf', bbox_inches='tight', dpi=4000, frameon=False)
    # fig.show()
    plt.close(fig)


num_data = 1
# Environment
Servers = 3
Channel_corr = 0.9
SNR = 10.0
pi = 320
W = 4
# Network
lr = 0.0001
df = 0.0
sl = 18
nhl = 1
eps_decay = 0.9
W_a = [3, 4]
lr_a = [0.1, 0.01, 0.001, 0.0001, 0.00001]  # learning rates
df_a = [0.0, 0.5, 0.9, 1]  # discount factor
# sl_a = [state_size, 2 * state_size, 5 * state_size, 10 * state_size]  # size of layer
nhl_a = [1, 4, 16]  # number of hidden layers
ef_a = [0.3, 0.9, 0.999, 0.9999]  # exploration factor
# paramter_2 = '_lr{}_df{}_sl{}_nhl{}_ef{}'.format(lr, df, sl, nhl, eps_decay)

# comb = list(product(nhl_a))
comb = list(range(2, 11))
# comb = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.999]
# comb = [3]
#comb = np.asarray(comb)
print(comb)

var = r'$\v' + 'arepsilon'
plot_AvgError = plot_desc(title='Error', x_label='Episodes', y_label=var + '_O$', x_axis=[0, 200], y_axis=[0, 1], legend=['RL'], marker='-')
plot_AbsError = plot_desc(title='Absolute Error', x_label='Cycles', y_label=var + '_O$', x_axis=[0, 200000], y_axis=[0, 1], legend=['RL'], marker='.')
plot_AvgLoss = plot_desc(title='Average Loss', x_label='Episodes', y_label='Average Loss', x_axis=[0, 200], y_axis=[0, 1], legend=['RL'], marker='-')
plot_AbsLoss = plot_desc(title='Absolute Loss', x_label='Cycles', y_label='Loss', x_axis=[0, 9984], y_axis=[0,1], legend=['RL'], marker='.')

# data = []
# for i in range(num_data):
#     data.append([])
index = 0
num = []
error_metric = []
loss_metric = []
error_last = []
error_best = []
loss_last = []
name = []

# Create Fig for 2-small subplots
fig, ax = plt.subplots(1, 2, figsize=(6, 2.5), dpi=1000)          # 2.214
# fig = plt.figure(figsize=(2.95, 2.214), dpi=1000)
# ax = fig.add_axes([0.26, 0.26, 0.7, 0.7])
#sum_sign_err = r'$\f' + 'rac{\sum' + r'\v' + 'arepsilon_O}' + '{M}$'
sum_sign_err = '$' + r'\v' + 'arepsilon_{O,log}$'
sum_sign_loss = '$L_{\log}$'
var = r'$\v' + 'arepsilon_{O}$'
rho = r'$\r' + 'ho$'
indices = [0]

opt_outdated = np.loadtxt('Optimal_OverK_DiffSNR.csv', delimiter=',')
for i in range(len(comb)):
    sl = comb[i]*6
    # parameters = 'DQN_S{}_rho{}_SNR{}_PS{}'. \
    #     format(comb[i], Channel_corr, SNR, pi)

    parameters = '_DQN__S{}_rho{}_SNR{}_PS{}_W1_lr{}_df{}_sl{}_nhl{}_ef{}_6'. \
        format(comb[i], Channel_corr, SNR, pi, lr, df, sl, nhl, eps_decay)

    print(parameters)
    # parameters = '_DQN_S{}_rho{}_SNR{}_PS{}_lr{}_df{}_sl{}_nhl{}_ef{}_6'.format(Servers, Channel_corr, SNR, pi, comb[i][0], comb[i][1], 18, int(comb[i][2]), comb[i][3])
    name_avg_error = 'Avg_Error' + parameters
    name_abs_error = 'Abs_Error' + parameters
    name_avg_loss = 'Avg_Loss' + parameters
    name_abs_loss = 'Abs_Loss' + parameters
    # plot_name = 'DQN Learning_FadingChannel_PerfectCSI_S{}'.format(comb[i])
    directory_load = 'FadingChannel_PerfectCSI/'
    directory_save_fig = 'FadingChannel_PerfectCSI/Figures/'

    avg_error = np.loadtxt(directory_load + name_avg_error + '.csv')

    abs_error = np.loadtxt(directory_load + name_abs_error + '.csv')
    avg_loss = np.loadtxt(directory_load + name_avg_loss + '.csv')
    abs_loss = np.loadtxt(directory_load + name_abs_loss + '.csv')

    # name.append(np.asarray(comb[i]))
    loss_log_avg = []
    abs_loss[abs_loss == 0] = 10**-50
    loss_log_avg.append(np.power(10, np.sum(- np.log10(abs_loss[:34]))/ len(abs_loss[:34])))
    subs = np.split(abs_loss[34:], 199)
    for d in subs:
        add = - np.log10(d)
        # print(d)
        # print(add)
        # print(sum(add))
        loss_log_avg.append(np.power(10, - np.sum(add)/ len(d)))
    # print(len(loss_log_avg))
    e_avg_b = min(avg_error)
    e_avg_l = avg_error[-1]
    l_avg_l = avg_loss[-1]
    e2000 = np.power(10, -np.sum(-np.log10(abs_error[-10000:])/len(abs_error[-10000:])))
    l500 = np.sum(abs_loss[-500:])/len(abs_loss[-500:])
    # print(len(avg_error[-10:]))
    # print(len(abs_loss))
    # print(data)
    # print(len(data))
    error_last.append(e_avg_l)
    error_best.append(e_avg_b)
    loss_last.append(l_avg_l)
    error_metric.append(e2000)
    loss_metric.append(l500)
    num.append(index)

    # # Plot
    print(opt_outdated[1])
    x_data = list(range(0, len(avg_error)))
    # alpha = 0.4 + index*0.2
    ax[0].plot(x_data, avg_error, color=petrol1, linestyle='-', linewidth=1)
    ax[0].plot(x_data, [np.transpose(opt_outdated)[3][1]]*len(x_data), color=(0.9609375, 0.65625, 0, 1), linestyle='dotted', linewidth=1)
    ax[1].plot(x_data, loss_log_avg, color=petrol1, linestyle='-', linewidth=1)
    index += 1

    # plot_data2(avg_error, [0, 200], [0, 10**-6], 'Episodes', sum_sign_err, plot_AvgError.title + '_' + parameters, 0,
    #            1, 1, directory_save_fig, '-', (0, 0.37890625, 0.39453125, 0.8), 0, 10**-9)
    # plot_data2(avg_loss, [0, 200], [0, 10**-1], 'Episodes', sum_sign, plot_AbsError.title + '_' + parameters, 0,
    #            1, 1, directory_save_fig, '-', (0, 0.37890625, 0.39453125, 0.8), 0, 10**-8)

    # plot_data_4box(np.transpose(avg_error), np.transpose(abs_error), np.transpose(avg_loss), np.transpose(abs_loss),
    #             x_axis=plot_AvgError.x_axis, y_axis=plot_AvgError.y_axis, x_label=plot_AvgError.x_label,
    #             y_label=plot_AvgError.y_label, title=plot_name,
    #             legend=plot_AvgError.legend, log=1, id=1, dir=directory_save_fig, marker=plot_AvgError.marker)
    # Plots
    # plot_data(np.transpose(avg_error), x_axis=plot_AvgError.x_axis, y_axis=plot_AvgError.y_axis, x_label=plot_AvgError.x_label,
    #           y_label=plot_AvgError.y_label, title=plot_AvgError.title + parameters,
    #           legend=plot_AvgError.legend, log=1, id=1, dir=directory_save_fig, marker=plot_AvgError.marker)
    # plot_data(np.transpose(abs_error), x_axis=plot_AbsError.x_axis, y_axis=plot_AbsError.y_axis,
    #           x_label=plot_AbsError.x_label,
    #           y_label=plot_AbsError.y_label, title=plot_AbsError.title + parameters, legend=plot_AbsError.legend, log=1,
    #           id=1, dir=directory_save_fig, marker=plot_AbsError.marker)
    # plot_data(np.transpose(avg_loss), x_axis=plot_AvgLoss.x_axis, y_axis=plot_AvgLoss.y_axis,
    #           x_label=plot_AvgLoss.x_label,
    #           y_label=plot_AvgLoss.y_label, title=plot_AvgLoss.title + parameters, legend=plot_AvgLoss.legend, log=1,
    #           id=1, dir=directory_save_fig, marker=plot_AvgLoss.marker)
    # plot_data(np.transpose(abs_loss), x_axis=plot_AbsLoss.x_axis, y_axis=plot_AbsLoss.y_axis,
    #           x_label=plot_AbsLoss.x_label, y_label=plot_AbsLoss.y_label, title=plot_AbsLoss.title + parameters,
    #           legend=plot_AbsLoss.legend, log=1, id=1, dir=directory_save_fig, marker=plot_AbsLoss.marker)

# Start 2-Small Subplots
# leg = fig.legend(['$N_{\mathrm{hl}}$=1', '$N_{\mathrm{hl}}$=4', '$N_{\mathrm{hl}}$=16'], facecolor='white', framealpha=1, fancybox=False,
#                  edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
# leg = fig.legend(['$' + r'\a' + 'lpha$=0.1', '$' + r'\a' + 'lpha$=0.01', '$' + r'\a' + 'lpha$=0.001', '$' + r'\a' + 'lpha$=0.0001', '$' + r'\a' + 'lpha$=0.00001'], facecolor='white', framealpha=1, fancybox=False,
#                   edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=5)
# leg = fig.legend(['$' + r'\g' + 'amma_{\mathrm{df}}$=0.0', '$' + r'\g' + 'amma_{\mathrm{df}}$=0.5', '$' + r'\g' + 'amma_{\mathrm{df}}$=0.9', '$' + r'\g' + 'amma_{\mathrm{df}}$=1'], facecolor='white', framealpha=1, fancybox=False,
#                    edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=4)
# leg = fig.legend(['$' + r'\e' + 'psilon_{\mathrm{dec}}$=0.3', '$' + r'\e' + 'psilon_{\mathrm{dec}}$=0.9', '$' + r'\e' + 'psilon_{\mathrm{dec}}$=0.999', '$' + r'\e' + 'psilon_{\mathrm{dec}}$=0.9999'], facecolor='white', framealpha=1, fancybox=False,
#                     edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=4)
# leg = fig.legend(['DQN-best', 'DQN-end', 'Optimal'], facecolor='white', framealpha=1, fancybox=False,
#                     edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=2)
# leg = fig.legend(['DQN', 'AS'], facecolor='white', framealpha=1, fancybox=False,
#                     edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.93), ncol=2)
# leg.get_frame().set_linewidth(0.1)
# # #
# for s in range(2):
#     ax[s].spines['right'].set_visible(False)
#     ax[s].spines['top'].set_visible(False)
#     ax[s].spines['left'].set_position(('axes', 0))
#     ax[s].spines['bottom'].set_position(('axes', 0))
#     ax[s].grid(True, linewidth=0.1, which='both')
#     ax[s].set_xlabel('$\mathcal{H}$')
#     ax[s].tick_params(which='minor', width=0.2)
#     ax[s].set_xlim(0, 200)
#
# ax[0].set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6, 8], linthreshy=10 ** -8)
# ax[1].set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6, 8], linthreshy=10 ** -10)
# ax[0].set_ylim(0, 10**-7)
# ax[1].set_ylim(0, 1)
# ax[0].set_ylabel(var)
# ax[1].set_ylabel(sum_sign_loss)
#
# fig.tight_layout()
# # plt.savefig(directory_save_fig + '04-Error_DQN_HP_nhl.eps', bbox_inches='tight', dpi=1000, frameon=False)
# plt.savefig(directory_save_fig + '04-Error_' + parameters + '.jpeg', bbox_inches='tight', dpi=1000, frameon=False)
#plt.show()
# # End Plot
#
# # print(len(loss_last))
# print(error_last)

# print(loss_last)
# print(np.argmin(np.asarray(loss_last)))
# print(name[np.argmin(np.asarray(loss_last))])
# np.savetxt('DQN Perf_Hyperparameters.csv', np.transpose([num, error_metric, loss_metric, error_last, loss_last, name]),
#            header='ID, Error, Loss, Error_Last, Loss_Last, LR_DF_NL_EF', fmt=['%1d', '0%30.28f', '0%30.28f', '0%30.28f', '0%30.28f', '%s'], delimiter=',')

# Large 1-Plot
# opt_all = np.loadtxt('Optimal_OverK_DiffSNR.csv', delimiter=',')
# opt_all = np.loadtxt('Error_Opt_Equal_overK_OCSI.csv', delimiter=',')
opt_all = np.loadtxt('Error_Opt_Equal_overK_pCSI.csv', delimiter=',')
opt_K3 = np.transpose(opt_all)
print(opt_K3)
# print(len(opt_K3))
print(error_last)
print(len(error_last))
fig = plt.figure(figsize=(5, 3.75), dpi=1000)
# fig = plt.figure(figsize=(5, 3.75), dpi=1000)
ax = fig.add_axes([0.05, 0.05, 0.85, 0.85])
# x_data = list(range(2, 11))

ax.grid(True, linewidth=0.1, which='both')
ax.plot(comb, error_last, color=petrol, linestyle='dashed', linewidth=2, marker='X',
        markersize=6, markerfacecolor=red, markeredgecolor=red, rasterized=True)
ax.plot(comb, error_best, color=petrol, linestyle='-', linewidth=3,
        marker='X', markersize=4, markerfacecolor=maigreen, markeredgecolor=maigreen, rasterized=True)
ax.plot(comb, opt_K3, color=orange, linestyle=(0, (1, 1)), linewidth=2, marker='o',
        markersize=2, markerfacecolor=blue,  markeredgecolor=blue, rasterized=True)
# ax.plot(comb, opt_K3[1], color=(0.9609375, 0.65625, 0, 0.65), linestyle='-', linewidth=1, rasterized=True) #  fading pCSI
# ax.plot(comb, opt_K3[2], color=(0.9609375, 0.65625, 0, 0.775), linestyle='-', linewidth=1, rasterized=True)                         # static
# ax.plot(comb, opt_K3[3], color=(0.9609375, 0.65625, 0, 0.875), linestyle='-', linewidth=1, rasterized=True)
# ax.plot(comb, opt_K3[4], color=(0.9609375, 0.65625, 0, 1), linestyle='-', linewidth=1, rasterized=True)

ax.spines['left'].set_position(('axes', 0))
ax.spines['bottom'].set_position(('axes', 0))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

leg = fig.legend(['DQN-Last', 'DQN-Best', 'AS'], facecolor='white', framealpha=1, fancybox=False, edgecolor='black',
                 loc='upper center', bbox_to_anchor=(0.535, 1), ncol=3)
leg.get_frame().set_linewidth(0.1)
ax.set_xlim(2, 10)
ax.set_ylim(0, 10**-3)
ax.tick_params(which='minor', width=0.2)
# ax.set_xlabel(rho)
ax.set_xlabel('$K$')
# ax.set_xlabel('Channel correlation factor' + rho)
# ax.set_ylabel('$' + r'\v' + 'arepsilon_{O}$')
ax.set_ylabel(sum_sign_err)
ax.set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6, 8], linthreshy=10**-7)

fig.tight_layout()
# plt.savefig(directory_save_fig + '03-OverK_fixedChannel_short_2' + '.pdf', bbox_inches='tight', dpi=500, frameon=False)
plt.savefig(directory_save_fig + '04-DQN_Fading_PCSI_OverK_slides' + '.jpeg', bbox_inches='tight', dpi=500, frameon=False)
# # plt.show()