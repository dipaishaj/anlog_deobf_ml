# STANDARD DECLARATIONS

import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
import numpy as np
import pandas as pd
import seaborn as sns

import PySpice.Logging.Logging as Logging
logger = Logging.setup_logging()

from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Probe.Plot import plot
from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit
from PySpice.Unit import *

import math
from engineering_notation import EngNumber

#####################################################################
font_size = 10
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                             "font.serif": ["Palatino"], "xtick.labelsize": font_size,
                             "ytick.labelsize": font_size, 'axes.grid': True,
                             'grid.linestyle': '--', 'lines.linewidth': 1,"axes.linewidth": 2,
                             'axes.axisbelow': True})
#####################################################################
# CIRCUIT NETLIST
circuit = Circuit('Step Response of RC Circuit')

#####################################################################
font_size = 14

plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (10, 5),
                     "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
                     'grid.linestyle': '--', 'lines.linewidth': 1.5, "axes.linewidth": 1.5, 'grid.color':'b', 'grid.linewidth': 1,
                     'axes.axisbelow': True})

steptime=1@u_us
switchingtime=20@u_ms
finaltime = 80@u_ms

capacitances = [0.1, 0.25, 0.5, 0.75, 1]

V1, V2, RES, CAP, Tau, VOUT, result, Vc = [], [], [], [], [], [], [], []
for c in capacitances:
    circuit = Circuit('Step Response of RC Circuit' + str(c))

    circuit.model('switch', 'SW', Ron=1@u_mΩ, Roff=1@u_GΩ)

    circuit.PulseVoltageSource(3, 'posa', circuit.gnd,initial_value=1, pulsed_value=-1,
                               pulse_width=finaltime, period=finaltime, delay_time=switchingtime)
    circuit.R('testa', 'posa', circuit.gnd, 1@u_kΩ)

    circuit.PulseVoltageSource(4, 'posb', circuit.gnd,initial_value=-1, pulsed_value=1,
                               pulse_width=finaltime, period=finaltime, delay_time=switchingtime)
    circuit.R('testb', 'posb', circuit.gnd, 1@u_kΩ)

    V_1 = circuit.V(1, 1, circuit.gnd, 40 @ u_V)
    circuit.R(1, 1, 'a', 20 @ u_kΩ)
    circuit.R(2, 'a', circuit.gnd, 60 @ u_kΩ)
    circuit.VoltageControlledSwitch(1, 'a', 'output', 'posa', circuit.gnd, model='switch')
    circuit.C(1, 'output', circuit.gnd, c @ u_uF)
    circuit.VoltageControlledSwitch(2, 'output', 'b', 'posb', circuit.gnd, model='switch')
    circuit.R(3, 'b', 2, 8 @ u_kΩ)
    circuit.R(4, 2, circuit.gnd, 120 @ u_kΩ)
    circuit.R(5, 2, 3, 40 @ u_kΩ)
    V_2 = circuit.V(2, circuit.gnd, 3, 20 @ u_V)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    ic=0@u_V
    simulator.initial_condition(output=ic)
    analysis = simulator.transient(step_time=steptime, end_time=finaltime)

    #######################################################################
    # THEORETICAL CALCULATIONS

    # Time constant after switching
    tau =(circuit.C1.capacitance)*(circuit.R3.resistance +
                                  (circuit.R4.resistance)*(circuit.R5.resistance)
                                  /(circuit.R4.resistance+ circuit.R5.resistance))
    print('tau={0}'.format(EngNumber(tau.value)))

    #######################################################################
    # CREATING DATAFRAME
    ######################################################################
    Eff_resi = (circuit.R3.resistance + (circuit.R4.resistance) * (circuit.R5.resistance)
                / (circuit.R4.resistance + circuit.R5.resistance))
    RES.append(Eff_resi)
    CAP.append(c)
    V1.append(int(40))
    V2.append(int(20))
    Tau.append(tau)
    vout = np.array(analysis["output"])
    # print("y_" + str(c), ": ",  vout[12000])
    Vout = [vout[x] for x in range(0, round(len(vout)/4), 1000)]
    Vc.append(vout[12000])
    VOUT.append(Vout)
    result.append((float(Eff_resi), float(c), tau))

    ######################################################################

    #######################################################################
    # PLOTTING COMMANDS

    figure = plt.subplots(figsize=(10, 8), sharex=True)

    axe = plt.subplot(311)
    plt.title('Switch activations', fontdict= {'fontsize': 20, 'fontweight':'bold'})
    #plt.xlabel('Time [s]')
    plt.ylabel('Voltage [V]', fontsize=18)
    plt.grid(visible=True)
    plot(analysis['posa'], axis=axe)
    pos_a = analysis['posa']
    plot(analysis['posb'], axis=axe)
    pos_b = analysis['posb']
    plt.tick_params(axis='both', which='both', labelsize=16)
    plt.legend(('position a', 'position b'), loc=(.05,.1))
    axe.set_yticks([-1, 0, 1])

    axe = plt.subplot(312)
    plt.title('Voltage across Capacitor', fontdict= {'fontsize': 20, 'fontweight':'bold'})
    plt.xlabel('Time [s]', fontsize=18)
    plt.ylabel('Voltage [V]', fontsize=18)
    plt.grid(visible=True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plot(analysis['output'], axis=axe)
    vout = analysis['output']
    # plt.legend(loc=(.05, .1))
    axe.set_yticks([-60, 0, 30])

# axe = plt.subplot(313, sharex=True)
my_array = np.array(VOUT)
cols = list(i for i in range(1, 22))
df2 = pd.DataFrame(my_array, columns= cols)
df1 = pd.DataFrame(result, columns=['R', 'C', 'Tau'])
ACsa_df = pd.concat([df1, df2], axis=1)
print("data : ", ACsa_df)
Sens_coeff = []

ACsa_df['C'].diff()
cap = ACsa_df['C'].diff().iloc[1]
print(cap)
newdf = ACsa_df.select_dtypes(include=np.number)
newdf.drop(['R', 'C'], axis=1, inplace=True)
new_df = abs(newdf.diff(axis=0).div(cap).fillna(0))
print(new_df)

# #######################################################################
# ## Sensitivity PLot ##
# # #######################################################################
new_df.columns.names = ['SAMPLE']
row = new_df.iloc[1]

fig = plt.figure(figsize=(10, 6))

axe = plt.subplot(313)
row.plot(kind='line', color='k')
plt.title('Sensitivity coefficients', fontdict= {'fontsize': 20, 'fontweight':'bold'})
plt.xlabel('Time [s]', fontsize=18)
plt.ylabel('Voltage [V]', fontsize=18)
plt.grid(visible=True)
plt.tick_params(axis='both', which='major', labelsize=16)
axe.set_yticks([-5, 0, 30])
axe.set_xticklabels([0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08])
plt.legend(('sensitivity'), loc=(.05,.1))
plt.tight_layout()
plt.show()


coef_mat =[]
for cols in new_df.columns:
    Sens_coeff.append(new_df[cols].div(cap))
print(Sens_coeff)


cov_matrix = ACsa_df.cov()
colormap = plt.cm.binary
f = plt.figure(figsize=(7, 7))
sns_heatmap = sns.heatmap(cov_matrix, square=True, cmap=colormap, linecolor='k', cbar=False,
                          linewidths=0.2, annot=True,
                          annot_kws={"fontsize":10}
                          # annot_kws={"size": 20 / np.sqrt(len(cov_matrix))}
                          )
# plt.title('Sensitivity coefficients between capacitor and output response values', fontdict= {'fontsize': 12, 'weight':'bold'})
plt.xlabel('Sample Number', fontsize=24)
plt.ylabel('Sensitivity coefficents', fontsize=24)
plt.tick_params(axis='both', which='both', labelsize=12)
plt.tight_layout()
f.savefig("/home/dips/PycharmProjects/pythonProject1/Figures/sens_heatmap.pdf")
f.savefig("/home/dips/PycharmProjects/pythonProject1/Figures/sens_heatmap.png")
plt.show()


