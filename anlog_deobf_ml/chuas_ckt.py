# Import all the necessary libraries.

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, SGDRegressor

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

from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import dataset_config
from Opamp import OpAmp
# ***********************************************************************
set_train_path = dataset_config.train_chua_ckt_path_2
# ***********************************************************************
## SET PARAMETERS FOR PLOTTING
font_size = 26
plt.rcParams.update({"text.usetex": True, "font.family": "serif",
                     "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (10, 5),
                     "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
                     'grid.linestyle': '--', 'lines.linewidth': 3.5, "axes.linewidth": 3.5, 'grid.color':'black', 'grid.linewidth': 1.5,
                     'axes.axisbelow': True})
# ***********************************************************************
### parameters
alpha = 15.395
beta = 28
# R = -1.143
# C_2 = -0.714
R = -1.143
C_2 = -0.714

def chua(u, t):
    x, y, z = u
    # electrical response of the nonlinear resistor
    f_x = C_2*x + 0.5*(R-C_2)*(abs(x+1)-abs(x-1))
    dudt = [alpha*(y-x-f_x), x - y + z, -beta * y]
    return dudt
# ***********************************************************************
# time discretization
t_0 = 0
dt = 1e-3
t_final = 300
t = np.arange(t_0, t_final, dt)

## initial conditions
u0 = [0.1,0,0]
## integrate ode system
sol = odeint(chua, u0, t)
# print(f'solution: {sol}')

#### plot eye diagram
## 3d-plot
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111, projection='3d', )
ax.set_xlabel('V1', fontsize=32)
ax.xaxis.set_ticklabels([])
ax.set_ylabel('V2', fontsize=32)
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])
# ax.set_zlabel('I_l')
#
ax.plot(sol[:,0],sol[:,1],sol[:,2], color='navy')
ax.set_title('Chaotic Circuit', fontsize=28)
plt.show()

# ***********************************************************************
## Gyrator circuit ########
class Gyrator(SubCircuit):
    NAME = 'GYRATOR'
    NODES = ('non_inv1', 'vp', 'vn')

    def __init__(self, name):
        SubCircuit.__init__(self, name, *self.NODES)

        self.subcircuit(OpAmp(name='OpAmp1'))
        self.X('op1', 'OpAmp1', 'non_inv1', 'inv1', 'vp', 'vn', 'out1')
        self.subcircuit(OpAmp(name='OpAmp2'))
        self.X('op2', 'OpAmp2', 'non_inv2', 'inv1', 'vp', 'vn', 'out2')

        self.R(7, 'non_inv1', 'out2', 100 @ u_Ohm)
        self.R(8, 'out2', 'inv1', 1 @ u_kOhm)
        self.R(9, 'inv1', 'out1', 1 @ u_kOhm)
        self.R(10, 'non_inv2',  self.gnd, 1.8 @ u_kOhm)
        self.C(10, 'non_inv2', 'out1', 100 @ u_nF)

        L = self.compute_L()

    def compute_L(self):
        eqv_L = (self['C10'].capacitance * self['R7'].resistance * self['R9'].resistance * self['R10'].resistance)/self['R8'].resistance
        # print(f'L:{eqv_L}')
        return eqv_L


class ChuaDiode(SubCircuit):
    NAME = 'Chua_Diode'
    NODES = ('non_inv', 'vp', 'vn')

    def __init__(self, name, Vsat=8.3 @u_V, Rc=1.8 @ u_kOhm, c1=10 @ u_nF, c2=100 @ u_nF, L=18 @ u_mH, G=(1/1.8e3)):
        SubCircuit.__init__(self, name, *self.NODES)
        self.Rc = Rc
        self.c1 = c1
        self.c2 = c2
        self.L = L
        self.G = G
        self.Vsat = Vsat
        self.g = None

        self.subcircuit(OpAmp(name='OpAmp1'))
        self.X('op1', 'OpAmp1', 'non_inv', 'e1', 'vp', 'vn', 'out3')
        self.subcircuit(OpAmp(name='OpAmp2'))
        self.X('op2', 'OpAmp2', 'non_inv', 'e2', 'vp', 'vn', 'out4')

        self.R(1, 'non_inv', 'out3', 220 @ u_Ohm)
        self.R(2, 'out3', 'e1', 220 @ u_Ohm)
        self.R(3, 'e1', self.gnd, 2.2 @ u_kOhm)
        self.R(4, 'non_inv', 'out4', 22 @ u_kOhm)
        self.R(5, 'out4', 'e2', 22 @ u_kOhm)
        self.R(6, 'e2', self.gnd, 3.3 @ u_kOhm)

    def compute_chua_g(self, u, t):
        x = u[0]
        y = u[1]
        z = u[2]
        E1 = (self['R3'].resistance / (self['R2'].resistance + self['R3'].resistance)) * self.Vsat
        E2 = (self['R6'].resistance / (self['R5'].resistance + self['R6'].resistance)) * self.Vsat

        m12 = - 1/self['R6'].resistance
        m02 = 1/self['R4'].resistance
        m01 = 1/self['R1'].resistance
        m11 = - 1 / self['R3'].resistance

        m1 = m12 + m11

        if E1 > E2:
            m0 = m11 + m02
        else:
            m0 = m12 + m01

        mm1 = m01 +m02

        Emax = np.max([E1, E2])
        Emin = np.min([E1, E2])

        # print(f'Emax :{Emax}')
        # print(f'Emin :{Emin}')

        if abs(x) < float(Emin):
            self.g = x * m1
        elif abs(x) < float(Emax):
            self.g = x * m0
            if x > 0:
                self.g = self.g + Emin * (m1 -m0)
            else:
                self.g = self.g + Emin * (m0 - m1)
        elif abs(x) >= float(Emax):
            self.g = x * mm1
            if x > 0:
                self.g = self.g + Emax * (m0 - mm1) + Emin * (m1 -m0)
            else:
                self.g = self.g + Emax * (mm1 - m0) + Emin * (m0 -m1)

        self.xdot = 1/(self.Rc *c1) * (G * (y - x) - self.g)
        self.ydot = 1 / (self.Rc * c2) * (G * (x - y) + z)
        self.zdot = - (1/L) * y

        # self.dudt()
        return [self.xdot, self.ydot, self.zdot]

    def dudt(self):
        return self.xdot, self.ydot, self.zdot


def check_len(arr):
    if len(arr) == 10:
        pass
    elif len(arr) > 10:
        l = len(arr)
        new_l = l - 10
        for i in range(new_l):
            del arr[i]
    elif len(arr) < 10:
        l = len(arr)
        new_l = 10 - l
        a = [0] * new_l
        arr.extend(a)

    return arr

# ***********************************************************************
# Circuit Simulation
nw_cnt = 0
Vout = []
len_v1 = 0
len_v2 = 0
len_ivdd = 0

circuit = Circuit('Chua')
Vop = 9 @ u_uV
Vsat = 6.3 @ u_uV

c1 = 10 @ u_nF
c2 = 100 @ u_nF
Rc = 1.7 @ u_kOhm

circuit.V('p', 'vdd', circuit.gnd, Vsat)
circuit.R('dd', 'vdd', 'vp', 700 @ u_Ohm)
circuit.V('n', 'vn', circuit.gnd,-Vsat)
circuit.V('op', 'vop', circuit.gnd, Vop)
circuit.V('on', 'von', circuit.gnd,-Vop)
circuit.R('c', 'v1', 'v2', Rc)
circuit.C(1, 'v1', circuit.gnd, c1)
circuit.C(2, 'v2', circuit.gnd, c2)

# circuit.R('o', 'vr', 'v2', 1 @ u_Ohm)
circuit.subcircuit(Gyrator(name='Gyr'))
circuit.X('Gy', 'Gyr', 'v2', 'vop', 'von')

gyr = Gyrator(SubCircuit)
L = gyr.compute_L()
print(f'L: {L}')

G = 1/circuit['Rc'].resistance
print(f'G: {G}')

circuit.subcircuit(ChuaDiode(name='Chua_D', Vsat=Vsat, c1=c1, c2=c2, L=L, G=G, Rc=Rc))
circuit.X('Dc', 'Chua_D', 'v1', 'vp', 'vn')

simulator = circuit.simulator(temperature=25, nominal_temperature=25)
analysis = simulator.transient(step_time=.001, end_time=.5)

V1 = np.array(analysis['v1'])
# print(len(V1))
V2 = np.array(analysis['v2'])
# print(f'V1: {V1}')
# print(f'V2: {V2}')

# *******************************************************************************
#  Create Dataset for Model Training
# *******************************************************************************
ln = np.array(analysis.time)
total_ln = len(ln)
# print(f'time: {ln}')
print(f'Total array_len : {total_ln}')

time_step = [t for t in range(0, 100, 1)]

v_1 = []
v_2 = []
i_vdd = []

new_v1 = []
new_v2 = []
new_vdd = []

for x in range(0, 100, 1):
    for time in time_step:
        v_1.append(float(analysis.v1[time+x]))
        v_2.append(float(analysis.v2[time+x]))
        i_vdd.append(float(analysis.vp[time+x]))
    v_1.extend(i_vdd)
    new_v1.extend([v_1])

for time in time_step:
    v_2.append(float(analysis.v2[time]))
v_2 = check_len(v_2)

for time in time_step:
    i_vdd.append(float(analysis.vp[time]))

i_vdd = check_len(i_vdd)

f = open(set_train_path + 'Chua_Ckt_%s.sp' % nw_cnt, 'w')
f.write("VOUT {}\n".format(Vout))
f.write(str(circuit))
f.close()
nw_cnt += 1

len_v1 = len(v_1)
len_v2 = len(v_2)
len_ivdd = len(i_vdd)

# # ***********************************************************
# Create DataFrame
# ***********************************************************
columns = ['v1_'+str(i) for i in range(0, int(len_v1/2))]
columns.extend(['ivdd_'+str(i) for i in range(0, int(len_v1/2))])

# columns.extend(['v2_'+str(i) for i in range(0, len_v2)])
df = pd.DataFrame(new_v1, columns=columns)
# #####################################################################################

data = pd.read_csv('./chua-LT1351.txt', sep="\t", header=None)
data.columns = ['time', 'V(v1)', 'V(v2)', 'I(R9)', 'I(R10)']

print(data.sample(5))

# generate x and y data for model
acc = []
X = df[['ivdd_'+str(i) for i in range(0, int(len_v1/2))]]
# Y = df[['ivdd_'+str(i) for i in range(1, 51, 10)]]
Y = df[['v1_99']]

std_scaler = StandardScaler()
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=15, shuffle=True)

X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)
y_train_std = std_scaler.fit_transform(y_train)
y_test_std = std_scaler.transform(y_test)

#
RF = RandomForestRegressor(n_estimators=200, max_features='sqrt', max_depth=15, random_state=9)
RF.fit(X_train_std, y_train_std)
#
y_pred = RF.predict(X_test_std)
y_predict = y_pred.reshape(-1, 1)

y_predict = std_scaler.inverse_transform(y_predict)
y_test = std_scaler.inverse_transform(y_test_std)
#
print(f' MAE: {mean_absolute_error(y_predict, y_test)}')
print(f' MSE: {mean_squared_error(y_predict, y_test)}')

y_predict = y_predict.ravel()
y_test = y_test.ravel()

for i in range(len(y_test)):
    pe = ((y_test[i] - y_predict[i])/y_test[i]) * 100
    acc.append(pe)

print(f' accuracy: {100 - np.mean(acc)}')
# print(f'prediction: {y_test}')

result = pd.DataFrame({'True': y_test, 'predict': y_predict})
result.to_csv('/home/dips/PycharmProjects/pythonProject1/data/chuas_0_200.csv')
plt.scatter(y_predict, y_test)
