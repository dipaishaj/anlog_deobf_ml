import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

pd.set_option("display.precision", 18)
import seaborn as sns
import random
import math

from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from PySpice.Plot.BodeDiagram import bode_diagram
from matplotlib.widgets import Cursor

import PySpice.Logging.Logging as Logging
from PySpice.Doc.ExampleTools import find_libraries

logger = Logging.setup_logging()

from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory

libraries_path = find_libraries()
spice_library = SpiceLibrary(libraries_path)
from PySpice.Unit import *


font_size = 20

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (10,8),
                     "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
                     'grid.linestyle': '--', 'lines.linewidth': 2.5, "axes.linewidth": 3.5, 'grid.color':'black', 'grid.linewidth': 1,
                     'axes.axisbelow': True})

def check_len(arr, length: int):
    if len(arr) == length:
        pass
    elif len(arr) > length:
        l = len(arr)
        new_l = l - length
        for i in range(new_l):
            del arr[i]
    elif len(arr) < length:
        l = len(arr)
        new_l = length - l
        a = [0] * new_l
        arr.extend(a)

    return arr

## ******************** 3*3 crossbar **************************

# columns = []
# for i in range(0, 30):
#     columns.append('vo1_'+str(i))
#     columns.append('vo2_' + str(i))
#     columns.append('vo3_' + str(i))
#     columns.append('Io_' + str(i))
# columns.extend(['v1', 'v2', 'v3', 'c1', 'c2', 'm11', 'm12', 'm13', 'm21', 'm22', 'm23', 'm31', 'm32', 'm33'])
# print(columns)
# df = pd.DataFrame(columns=columns)
#
# cnt = 0
# time_step = []
# ton = 125 @ u_ms
# on_dealy = 1 @ u_us
# period = 250 @ u_ms
# c1 = 10e-1
# c2 = 10e-2
#
# for i in range(600):
#     m11, m21, m31 = np.random.uniform(200, 1000), np.random.uniform(200, 1000), np.random.uniform(200, 1000)
#     m12, m22, m32 = np.random.uniform(200, 1000), np.random.uniform(200, 1000), np.random.uniform(200, 1000)
#     m13, m23, m33 = np.random.uniform(200, 1000), np.random.uniform(200,1000), np.random.uniform(200, 1000)
#     for v1 in np.arange(1.5, 4.5, 1):
#         for v2 in np.arange(1.5, 4.5, 1):
#             for v3 in np.arange(2.5, 5.5, 1):
#                 circuit = Circuit('crossbar')
#                 # vn1 = circuit.PulseVoltageSource('n1', 'vn1', circuit.gnd, initial_value=1, pulsed_value=v1,
#                 #                                  pulse_width=ton, period=period, delay_time=on_dealy)
#                 # vn2 = circuit.PulseVoltageSource('n2', 'vn2', circuit.gnd, initial_value=0, pulsed_value=v2,
#                 #                                  pulse_width=ton, period=period, delay_time=on_dealy)
#                 # vn3 = circuit.PulseVoltageSource('n3', 'vn3', circuit.gnd, initial_value=0, pulsed_value=v3,
#                 #                                  pulse_width=ton, period=period, delay_time=on_dealy)
#                 vn1 = circuit.V('n1', 'vn1', circuit.gnd, v1 @ u_V)
#                 vn2 = circuit.V('n2', 'vn2', circuit.gnd, v2 @ u_V)
#                 vn3 = circuit.V('n3', 'vn3', circuit.gnd, v3 @ u_V)
#
#                 rs = 2 @ u_kOhm
#                 rn = 2 @ u_kOhm
#                 rw = 1 @ u_Ohm
#                 total_i = []
#
#                 circuit.R('sr1', 'vn1', 1, rs)
#                 circuit.R('sr2', 'vn2', 2, rs)
#                 circuit.R('sr3', 'vn3', 3, rs)
#
#                 circuit.R('w1', 1, 'n1', rw)
#                 circuit.R('w2', 2, 'n2', rw)
#                 circuit.R('w3', 3, 'n3', rw)
#
#                 circuit.R('m11', 'n1', 'm1', m11 @ u_Ohm)
#                 circuit.C('m11', 'n1', 'm1', 10 @ u_nF)
#                 circuit.R('m21', 'n2', 'm2', m21 @ u_Ohm)
#                 circuit.C('m21', 'n2', 'm2', 10 @ u_nF)
#                 circuit.R('m31', 'n3', 'm3', m31 @ u_Ohm)
#                 circuit.C('m31', 'n3', 'm3', 10 @ u_nF)
#
#                 circuit.R('w4', 'm1', 'n2', rw)
#                 circuit.R('w5', 'm2', 'n3', rw)
#                 circuit.R('w6', 'm3', 'o1', rw)
#
#                 circuit.R('w7', 'n1', 'n4', rw)
#                 circuit.R('w8', 'n2', 'n5', rw)
#                 circuit.R('w9', 'n3', 'n6', rw)
#
#                 circuit.R('m12', 'n4', 'm4', m12 @ u_Ohm)
#                 circuit.C('m12', 'n4', 'm4', 10 @ u_pF)
#                 circuit.R('m22', 'n5', 'm5', m22 @ u_Ohm)
#                 circuit.C('m22', 'n5', 'm5', 10 @ u_pF)
#                 circuit.R('m32', 'n6', 'm6', m32 @ u_Ohm)
#                 circuit.C('m32', 'n6', 'm6', 10 @ u_pF)
#
#                 circuit.R('w10', 'm4', 'n5', rw)
#                 circuit.R('w11', 'm5', 'n6', rw)
#                 circuit.R('w12', 'm6', 'o2', rw)
#
#                 circuit.R('w13', 'n4', 'n7', rw)
#                 circuit.R('w14', 'n5', 'n8', rw)
#                 circuit.R('w15', 'n6', 'n9', rw)
#
#                 circuit.R('m13', 'n7', 'm7', m13 @ u_Ohm)
#                 circuit.C('m13', 'n7', 'm7', 10 @ u_pF)
#                 circuit.R('m23', 'n8', 'm8', m23 @ u_Ohm)
#                 circuit.C('m23', 'n8', 'm8', 10 @ u_pF)
#                 circuit.R('m33', 'n9', 'm9', m33 @ u_Ohm)
#                 circuit.C('m33', 'n9', 'm9', 10 @ u_pF)
#
#                 circuit.R('w16', 'm7', 'n8', rw)
#                 circuit.R('w17', 'm8', 'n9', rw)
#                 circuit.R('w18', 'm9', 'o3', rw)
#
#                 circuit.R('n1', 'o1', circuit.gnd, rn)
#                 circuit.R('n2', 'o2', circuit.gnd, rn)
#                 circuit.R('n3', 'o3', circuit.gnd, rn)
#
#                 simulator = circuit.simulator(temperature=25, nominal_temperature=25)
#                 # analysis = simulator.operating_point()
#                 analysis = simulator.transient(step_time=0.01, end_time=10)
#
#                 vo1 = np.array(analysis['o1'])
#                 vo2 = np.array(analysis['o2'])
#                 vo3 = np.array(analysis['o3'])
#
#                 # print(np.array(analysis['o1']))
#                 # print(np.array(analysis['o2']))
#
#                 Isum = np.add(vo1, vo2, vo3)
#                 # print(f'sum of I : {Isum} A')
#                 print(len(Isum))
#
#                 total_ln = len(Isum)
#                 if total_ln >= 1800:
#                     time_step = [t for t in range(0, total_ln, 60)]
#                 elif 1800 > total_ln > 1700:
#                     time_step = [t for t in range(0, total_ln, 55)]
#                 elif 1700 >= total_ln > 1600:
#                     time_step = [t for t in range(0, total_ln, 54)]
#                 elif 1600 >= total_ln > 1500:
#                     time_step = [t for t in range(0, total_ln, 50)]
#                 elif 1500 >= total_ln > 1200:
#                     time_step = [t for t in range(0, total_ln, 40)]
#                 elif 1200 >= total_ln:
#                     time_step = [t for t in range(0, total_ln, 33)]
#
#                 time_step = check_len(time_step, 30)
#                 # print(f'len of time: {len(time_step)}')
#
#                 for t in time_step:
#                     total_i.append(vo1[t])
#                     total_i.append(vo2[t])
#                     total_i.append(vo3[t])
#                     total_i.append(Isum[t])
#
#                 total_i.extend([v1, v2, v3, c1, c2, m11, m12, m13, m21, m22, m23, m31, m32, m33])
#
#                 df.loc[len(df)] = total_i
#
#                 cnt += 1
#
# print(f'cnt: {cnt}')
# print(df.sample(5))
# df.to_csv('/home/dips/PycharmProjects/pythonProject1/data/crossbar_3by3_DC.csv')


result = pd.read_csv('/home/dips/PycharmProjects/pythonProject1/data/crossbar_prediction_3by3_ac.csv')
result = result.iloc[:200,:]

fig,ax = plt.subplots(3,3, figsize=(10,10))
fig.supxlabel('True Value', fontsize=34, fontweight='bold')
fig.supylabel('Predicted Value', fontsize=34, fontweight='bold')

ax[0, 0].scatter(x='m11_true', y='m11_predict', data=result, c='navy', s=7)
ax[0, 0].set_title('UK1', fontsize=26)
# ax[0, 0].tick_params(axis=y, labelcolor='k', size=14)
ax[0, 0].grid(linestyle='--')
# ax[0, 1].yaxis.set_ticklabels([])
ax[0, 0].xaxis.set_ticklabels([])

ax[0, 1].scatter(x='m12_true', y='m12_predict', data=result, c='magenta', s=7)
ax[0, 1].set_title('UK2', fontsize=24)
ax[0, 1].grid(linestyle='--')
ax[0, 1].yaxis.set_ticklabels([])
ax[0, 1].xaxis.set_ticklabels([])

ax[0, 2].scatter(x='m13_true', y='m13_predict', data=result, c='darkgreen', s=7)
ax[0, 2].set_title('UK3', fontsize=24)
ax[0, 2].grid(linestyle='--')
ax[0, 2].yaxis.set_ticklabels([])
ax[0, 2].xaxis.set_ticklabels([])

ax[1, 0].scatter(x='m21_true', y='m21_predict', data=result, c='black', s=7)
ax[1, 0].set_title('UK4', fontsize=24)
ax[1, 0].grid(linestyle='--')
# ax[1, 0].yaxis.set_ticklabels([])
ax[1, 0].xaxis.set_ticklabels([])

ax[1, 1].scatter(x='m22_true', y='m22_predict', data=result, c='coral', s=7)
ax[1, 1].set_title('UK5', fontsize=24)
ax[1, 1].grid(linestyle='--')
ax[1, 1].yaxis.set_ticklabels([])
ax[1, 1].xaxis.set_ticklabels([])

ax[1, 2].scatter(x='m23_true', y='m23_predict', data=result, c='red', s=7)
ax[1, 2].set_title('UK6', fontsize=24)
ax[1, 2].grid(linestyle='--')
ax[1, 2].yaxis.set_ticklabels([])
ax[1, 2].xaxis.set_ticklabels([])

ax[2, 0].scatter(x='m31_true', y='m31_predict', data=result, c='darkred', s=7)
ax[2, 0].set_title('UK7', fontsize=24)
ax[2, 0].grid(linestyle='--')
# ax[2, 0].yaxis.set_ticklabels([])
# ax[2, 0].xaxis.set_ticklabels([])

ax[2, 1].scatter(x='m32_true', y='m32_predict', data=result, c='blue', s=7)
ax[2, 1].set_title('UK8', fontsize=24)
ax[2, 1].grid(linestyle='--')
ax[2, 1].yaxis.set_ticklabels([])
# ax[0, 1].xaxis.set_ticklabels([])

ax[2, 2].scatter(x='m33_true', y='m33_predict', data=result, c='purple', s=7)
ax[2, 2].set_title('UK9', fontsize=24)
ax[2, 2].grid(linestyle='--')
ax[2, 2].yaxis.set_ticklabels([])
# ax[2, 2].xaxis.set_ticklabels([])

plt.savefig("/home/dips/PycharmProjects/pythonProject1/Figures/crossbar_3by3_ac.pdf")
plt.savefig("/home/dips/PycharmProjects/pythonProject1/Figures/crossbar_3by3_ac.png")

fig.tight_layout()
fig.subplots_adjust(right=0.97)
plt.show()
