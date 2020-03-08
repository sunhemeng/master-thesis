import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from matplotlib.ticker import MultipleLocator
from tikzplotlib import save as tikz_save

'''
Plot error, loss and gradient of data files
'''


# plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
# lsfont = {'fontname':'lmodern'}
plt.rcParams['text.usetex'] = True
# plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']
plt.rcParams["font.family"] = ["lmodern"]
plt.rcParams.update({"font.size": 10})
plt.rcParams.update({"legend.fontsize": 8})

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
color_array = [red, hellred, hellred2]  # lr
# color_array_grad = [[(0.796875, 0.02734375, 0.1171875, 1), (0.796875, 0.02734375, 0.1171875, 0.7)],
#                [(0.84375, 0.359375, 0.253906425, 1), (0.84375, 0.359375, 0.253906425, 0.85)],
#                [(0.8984375, 0.5859375, 0.47265625, 1), (0.8984375, 0.5859375, 0.47265625, 0.85)]] # lr
# color_array = [darkred, 'red', hellwinered, hellwinered2, ]  # df
# color_array = [violett, violett2, hellviolett, hellviolett2, ]  # sigma
# color_array = [hellmagenta, lila, helllila2]  # nhl

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
    plt.figure(fig_no, figsize=(2.85, 2.3), dpi=2000)
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
lr = 0.00005
df = 0.0
nhl = 1

sigma = 0.8
theta = 0.3
lr_a = [0.0005, 0.00005, 0.000005]  # learning rates
df_a = [0.0, 0.3, 0.9, 1]  # discount factor
# sl_a = [state_size, 2 * state_size, 5 * state_size, 10 * state_size]  # size of layer
nhl_a = [1, 4, 8]  # number of hidden layers
ef_a = [0.9, 0.999, 0.9999]  # exploration factor
# paramter_2 = '_lr{}_df{}_sl{}_nhl{}_ef{}'.format(lr, df, sl, nhl, eps_decay)
# sigma_a = [0.2, 0.5, 0.8, 1]
# comb = list(product(lr_a))
# comb = range(2,8)
# comb = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
comb = [3]
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
# # fig = plt.figure(figsize=(2.95, 2.214), dpi=1000)
# # ax = fig.add_axes([0.26, 0.26, 0.7, 0.7])
#sum_sign_err = r'$\f' + 'rac{\sum' + r'\v' + 'arepsilon_O}' + '{M}$'
sum_sign_err = '$' + r'\v' + 'arepsilon_{O,log}$'
sum_sign_loss = '$L_{log}$'
sum_grad = '$' + r'\f' + 'rac{1}{U_{\mathrm{b}}}\sum_{i}' + r'\n' + 'abla_{b} Q(s, b| ' + r'\t' + 'heta^{Q})$'
var = r'$\v' + 'arepsilon$'
rho = r'$\r' + 'ho$'
indices = [0]

opt_outdated = np.loadtxt('Optimal_OverK_DiffSNR.csv', delimiter=',')
# print(np.transpose(opt_outdated)[2])
for i in range(len(comb)):
    # # sl = comb[i]*6
    parameters = '_S{}_rho{}_SNR{}_PS{}'. \
        format(comb[i], Channel_corr, SNR, pi)
    # parameters = '_DDPG_S{}_rho{}_SNR{}_PS{}_lr{}_df{}_sigOU{}_thetaOU{}'. \
    #     format(Servers, Channel_corr, SNR, pi, lr, df, sigma, theta)

    print(parameters)
    # parameters = '_DQN_S{}_rho{}_SNR{}_PS{}_lr{}_df{}_sl{}_nhl{}_ef{}_6'.format(Servers, Channel_corr, SNR, pi, comb[i][0], comb[i][1], 18, int(comb[i][2]), comb[i][3])
    name_avg_error = 'Error' + parameters
    name_abs_error = 'Abs_Error' + parameters
    name_avg_grad = 'AvgGradient' + parameters
    # name_avg_loss = 'AvgLoss_Critic' + parameters
    # name_abs_loss = 'Abs_Loss' + parameters
    # plot_name = 'DQN Learning_FadingChannel_PerfectCSI_S{}'.format(comb[i])
    directory_load = 'FadingChannel_OutdatedCSI/'
    directory_save_fig = 'FadingChannel_OutdatedCSI/Figures/'

    avg_error = np.loadtxt(directory_load + name_avg_error + '.csv')
    #abs_error = np.loadtxt(directory_load + name_abs_error + '.csv')
    avg_grad = np.transpose(np.loadtxt(directory_load + name_avg_grad + '.csv'))
    print(avg_grad[0])
    # avg_loss = np.transpose(np.loadtxt(directory_load + name_avg_loss + '.csv'))
    # abs_loss = np.loadtxt(directory_load + name_abs_loss + '.csv')

    # name.append(np.asarray(comb[i]))
    # loss_log_avg = []
    # avg_loss[avg_loss == 0] = 10**-50
    # loss_log_avg.append(np.power(10, np.sum(- np.log10(avg_loss[:488]))/ len(avg_loss[:488])))
    # subs = np.split(avg_loss[488:], 398)
    # for d in subs:
    #     add = - np.log10(d)
    #     # print(d)
    #     # print(add)
    #     # print(sum(add))
    #     loss_log_avg.append(np.power(10, - np.sum(add)/ len(d)))
    # # print(len(loss_log_avg))
    e_avg_b = min(avg_error)
    e_avg_l = avg_error[-1]
    # l_avg_l = avg_loss[-1]

    # e2000 = np.power(10, -np.sum(-np.log10(abs_error[-10000:])/len(abs_error[-10000:])))
    # l500 = np.sum(abs_loss[-500:])/len(abs_loss[-500:])
    # print(len(avg_error[-10:]))
    # print(len(abs_loss))
    # print(data)
    # print(len(data))
    error_last.append(e_avg_l)
    error_best.append(e_avg_b)
    # loss_last.append(l_avg_l)
    #error_metric.append(e2000)
    # loss_metric.append(l500)
    # num.append(index)
    # print(avg_grad)
    # print(len(avg_grad[0]))
    # print(len(avg_grad[1]))
    # Plot
    # print(len(loss_log_avg))
    x_data = list(range(0, len(avg_error)))
    # # print(len(x_data))
    # x2_data = list(range(0, len(avg_grad)))
    # alpha = 0.3 + index*0.2
    ax[0].plot(x_data, avg_error, color=hellred, linestyle='-', linewidth=1)
    ax[0].plot(x_data, [np.transpose(opt_outdated)[3][1]]*len(x_data), color=(0.9609375, 0.65625, 0, 1), linestyle='--', linewidth=1)
    ax[1].plot(avg_grad[0], color=hellviolett, linestyle='-', linewidth=0.1)
    ax[1].plot(avg_grad[1], color=helllila, linestyle='-', linewidth=0.1)
    # ax[1].plot(avg_grad[0], color=color_array_grad[i][0], linestyle='-', linewidth=0.1)
    # ax[1].plot(avg_grad[1], color=color_array_grad[i][1], linestyle='-', linewidth=0.1)
    # ax[1].plot(x_data, loss_log_avg, color=color_array[i], linestyle='-', linewidth=1)
    #ax[1].plot(x2_data, avg_loss[1], color=violett, linestyle='-', linewidth=0.1)
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
# leg = fig.legend(['$N_{\mathrm{hl}}$=1', '$N_{\mathrm{hl}}$=4', '$N_{\mathrm{hl}}$=8'], facecolor='white', framealpha=1, fancybox=False,
#                  edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
# leg = fig.legend(['$' + r'\a' + 'lpha$=0.0005', '$' + r'\a' + 'lpha$=0.00005', '$' + r'\a' + 'lpha$=0.000005'], facecolor='white', framealpha=1, fancybox=False,
#                   edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
# leg = fig.legend(['$' + r'\g' + 'amma_{\mathrm{df}}$=0.0', '$' + r'\g' + 'amma_{\mathrm{df}}$=0.5', '$' + r'\g' + 'amma_{\mathrm{df}}$=0.9', '$' + r'\g' + 'amma_{\mathrm{df}}$=1'], facecolor='white', framealpha=1, fancybox=False,
#                    edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=4)
# leg = fig.legend(['$' + r'\e' + 'psilon_{dec}$=0.9', '$' + r'\e' + 'psilon_{dec}$=0.999', '$' + r'\e' + 'psilon_{dec}$=0.9990'], facecolor='white', framealpha=1, fancybox=False,
#                     edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=4)
# leg = fig.legend([r'$\s' + 'igma_{\mathrm{OU}}$=0.2', r'$\s' + 'igma_{\mathrm{OU}}$=0.5', r'$\s' + 'igma_{\mathrm{OU}}$=0.8',  r'$\s' + 'igma_{\mathrm{OU}}$=1'], facecolor='white', framealpha=1, fancybox=False,
#                      edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=4)
# leg = fig.legend(['DQN-best', 'DQN-end'], facecolor='white', framealpha=1, fancybox=False,
#                     edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=2)
leg = fig.legend(['DDPG', 'AS', '$b_1$', '$b_2$'], facecolor='white', framealpha=1, fancybox=False,
                    edgecolor='black', loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=4)
leg.get_frame().set_linewidth(0.1)
#
for s in range(2):
    ax[s].spines['right'].set_visible(False)
    ax[s].spines['top'].set_visible(False)
    ax[s].spines['left'].set_position(('axes', 0))
    ax[s].spines['bottom'].set_position(('axes', 0))
    ax[s].grid(True, linewidth=0.1, which='both')
    # ax[s].set_xlabel('Episodes')
    ax[s].tick_params(which='minor', width=0.2)
ax[1].ticklabel_format(axis='x', style='scientific')
ax[0].set_xlabel('$\mathcal{H}$')
# ax[1].set_xlabel('$\mathcal{H}$')
ax[1].set_xlabel('$M_{\mathrm{train}}$')
ax[0].set_xlim(0, 400)
ax[1].set_xlim(0, 400000)
ax[0].set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6, 8], linthreshy=10 ** -8)
ax[1].set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6, 8], linthreshy=10 ** -5)
ax[0].set_ylim(0, 1)
ax[1].set_ylim(0, 10)
ax[0].set_ylabel(sum_sign_err)
ax[1].set_ylabel(sum_grad)
# leg = fig.legend([var, '$A_1$', '$A_2$'], facecolor='white', framealpha=1, fancybox=False, edgecolor='black',
#                  loc='lower center', bbox_to_anchor=(0.5, 0.9), ncol=3)
# leg.get_frame().set_linewidth(0.1)
for line in leg.get_lines():
    line.set_linewidth(1)
fig.tight_layout()
plt.savefig(directory_save_fig + '05-DDPG_Outdated_' + parameters + '.pdf', bbox_inches='tight', dpi=1000, frameon=False)
# plt.savefig(directory_save_fig + '05-DDPG_OCSI_HP_lr.eps', bbox_inches='tight', dpi=1000, frameon=False)
# plt.savefig(directory_save_fig + '05-DDPG_Static_Learning_K3.pdf', bbox_inches='tight', dpi=1000, frameon=False)
# plt.show()
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
# opt_all = np.loadtxt('05-Error_Opt_overRho_OCSI.csv', delimiter=',')
# opt_all = np.loadtxt('05-Error_Opt_overRho_OCSI.csv', delimiter=',')
# # opt_all = np.loadtxt('Optimal_OverK_DiffSNR.csv', delimiter=',')
# opt_K3 = np.transpose(opt_all)
# # print(opt_K3)
# # print(error_last)
# # print(len(error_last))
# # fig = plt.figure(figsize=(5, 3.75), dpi=1000)
# fig = plt.figure(figsize=(5, 3.75), dpi=1000)
# ax = fig.add_axes([0.05, 0.05, 0.85, 0.85])
# # x_data = list(range(2, 8))
# ax.grid(True, linewidth=0.1, which='both')
# print(len(opt_K3[1:-1]))
#
# # ax.plot(x_data, opt_K3, color=(0.9609375, 0.65625, 0, 0.9), linestyle=(0, (1, 1)), linewidth=1.25, marker='v', markersize=2, rasterized=True) #fading pCSI
# ax.plot(comb, error_last, color=(0.62890625, 0.0625, 0.20703125, 0.9), linestyle='dashed',
#         linewidth=2, marker='^', markersize=3, markerfacecolor=blue,  markeredgecolor=petrol, rasterized=True)
# ax.plot(comb, error_best, color=(0.62890625, 0.0625, 0.20703125, 0.9), linestyle='dotted',
#         linewidth=2, marker='v', markersize=3, markerfacecolor=blue, markeredgecolor=petrol, rasterized=True)
# ax.plot(comb, opt_K3[1:-1], color=orange, linestyle=(0, (1, 1)), linewidth=2, marker='o',
#         markersize=2, markerfacecolor=black,  markeredgecolor=black, rasterized=True)                         # static
# # ax.plot(x_data, error_last, color=petrol, linestyle='dashed', linewidth=1, marker='>', markersize=1.25, rasterized=True)
# # ax.plot(x_data, error_best, color=petrol, linestyle='dotted', linewidth=1, marker='<', markersize=1.25, rasterized=True)
#
# ax.spines['left'].set_position(('axes', 0))
# ax.spines['bottom'].set_position(('axes', 0))
# ax.spines['right'].set_visible(False)
# ax.spines['top'].set_visible(False)
#
# leg = fig.legend(['DDPG-Last', 'DDPG-Best', 'AS'], facecolor='white', framealpha=1, fancybox=False, edgecolor='black',
#                  loc='upper center', bbox_to_anchor=(0.535, 1), ncol=3)
#
# leg.get_frame().set_linewidth(0.1)
# ax.set_xlim(0.1, 0.9)
# ax.set_ylim(0, 10**-1)
# ax.tick_params(which='minor', width=0.2)
# # ax.set_xlabel('$K$')
# ax.set_xlabel(rho)
# # ax.set_xlabel('Channel correlation factor' + rho)
# # ax.set_ylabel('$' + r'\v' + 'arepsilon_{O}$')
# ax.set_ylabel(sum_sign_err)
# ax.set_yscale('symlog', nonposy='clip', subsy=[2, 4, 6, 8], linthreshy=10**-5)
#
# fig.tight_layout()
# # plt.savefig(directory_save_fig + '05-DDPG_OverK_OCSI_slides_long' + '.jpeg', bbox_inches='tight', dpi=1000, frameon=False)
# plt.savefig(directory_save_fig + '05-DDPG_OverRho_OCSI_slides_long' + '.pdf', bbox_inches='tight', dpi=1000, frameon=False)
# plt.show()