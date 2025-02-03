# *************************************************************************************
import math
import random

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import PySpice
import warnings
import PySpice.Logging.Logging as Logging
from PySpice.Doc.ExampleTools import find_libraries
from PySpice.Plot.BodeDiagram import bode_diagram

logger = Logging.setup_logging()

from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory

libraries_path = find_libraries()
spice_library = SpiceLibrary(libraries_path)
from PySpice.Unit import *

import dataset_config
from MillerOTA import MillerOTA
set_train_path = dataset_config.train_ota_bpf_path_4
# *************************************************************************************
# ****************************************************************************************
font_size = 10
rc = {"text.usetex": True, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (5, 4),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 1, 'lines.linewidth': 2.5, "axes.linewidth": 2.5,
          'axes.axisbelow': True}
plt.rcParams.update(rc)
# ****************************************************************************************
# ___________________________________________________________
# 'Forth-order Gm-C bandpass filter using OTA'
# ___________________________________________________________
nw_cnt = 1
mos_width = []
widths = []
m_w = []
for i in range(250):
    wd = random.uniform(1, 120.5)
    mos_width.append(wd)
for i in range(450):
    widths = random.sample(mos_width, 6)
    m_w.append(widths)

for wd in range(len(m_w)):
    w1, w2, w3, w4, w5, w6 = m_w[wd][0], m_w[wd][1], m_w[wd][2], m_w[wd][3], m_w[wd][4], m_w[wd][5]

    circuit = Circuit('OTA_BPF_4TH_' + str(nw_cnt))
    VSS = circuit.V('ss', 'Vss', circuit.gnd, -2.5 @ u_V)
    VDD = circuit.V('dd', 'Vdd', circuit.gnd, 2.5 @ u_V)
    Vin = circuit.SinusoidalVoltageSource('in', 'x0', circuit.gnd, amplitude=1 @ u_V, frequency=100 @ u_Hz)

    circuit.subcircuit(MillerOTA(name='OTA1',num_unk=1, wd2=w1))
    circuit.X('ota1', 'OTA1', 'x0', circuit.gnd, 'Vdd', 'Vss', 'out1')
    C11 = circuit.C(1, 'out1', circuit.gnd, 78.95@ u_pF)
    circuit.subcircuit(MillerOTA(name='OTA3', num_unk=3, wd2=w3))
    circuit.X('ota3', 'OTA3', circuit.gnd, 'out1', 'Vdd', 'Vss', 'out3')
    circuit.R('s11', 'out1', 'out3', 1@u_Ohm)
    circuit.subcircuit(MillerOTA(name='OTA2',num_unk=2, wd2=w2))
    circuit.X('ota2', 'OTA2', 'out3', circuit.gnd,  'Vdd', 'Vss', 'out2')
    C12 = circuit.C(2, 'out2', circuit.gnd, 78.95@u_pF)
    circuit.subcircuit(MillerOTA(name='OTA4', num_unk=0))
    circuit.X('ota4', 'OTA4', circuit.gnd, 'out2', 'Vdd', 'Vss', 'out4')
    circuit.R('s12', 'out4', 'out1', 1@u_Ohm)
    circuit.subcircuit(MillerOTA(name='OTA5', num_unk=4, wd2=w4))
    circuit.X('ota5', 'OTA5', 'out3', circuit.gnd, 'Vdd', 'Vss', 'out5')
    C21 = circuit.C(3, 'out5', circuit.gnd, 51.34 @ u_pF)
    circuit.subcircuit(MillerOTA(name='OTA7', num_unk=0))
    circuit.X('ota7', 'OTA7', circuit.gnd, 'out5', 'Vdd', 'Vss', 'y0')
    circuit.R('s21', 'out5', 'y0', 1@u_Ohm)
    circuit.subcircuit(MillerOTA(name='OTA6', num_unk=5, wd2=w5))
    circuit.X('ota6', 'OTA6', 'y0', circuit.gnd,  'Vdd', 'Vss', 'out6')
    C22 = circuit.C(4, 'out6', circuit.gnd, 51.34@u_pF)
    circuit.subcircuit(MillerOTA(name='OTA8', num_unk=6, wd2=w6))
    circuit.X('ota8', 'OTA8', circuit.gnd, 'out6', 'Vdd', 'Vss', 'out8')
    circuit.R('s22', 'out8', 'out5', 1@u_Ohm)
    # print(circuit)

    simulator = circuit.simulator(temperature=25, nominal_temperature=25)
    analysis = simulator.ac(start_frequency=100 @ u_Hz, stop_frequency=1 @ u_GHz, number_of_points=10,
                            variation='dec')

    vout = np.array(abs(analysis['out3']))
        # print(len(vout))
    Vout = [vout[x] for x in range(0, len(vout), 16)]
    for i in range(len(Vout)):
        Vout[i] = float('{}'.format(Vout[i]))
    print("VOUT :", Vout)

    f = open(set_train_path + 'OTA_BPF_4TH_%s.sp' % nw_cnt, 'w')
    f.write("VOUT {}\n".format(Vout))
    f.write(str(circuit))
    f.close()
    nw_cnt += 1
# ****************************************************************************************
## Plot Frequency response
# frequency = analysis.frequency
# gain = 20 * np.log10(np.absolute(analysis.y0))  # Convert magnitude to dB
# phase = np.degrees(np.angle(analysis.y0))  # Convert phase from radians to degrees
#
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
#
# # Plot gain
# ax1.semilogx(frequency, gain, marker='.', color='blue', linestyle='-')
# ax1.set_title('Bode Plot of the System')
# ax1.set_ylabel('Gain (dB)')
# ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
#
# # Plot phase
# ax2.semilogx(frequency, phase, marker='.', color='blue', linestyle='-')
# ax2.set_xlabel('Frequency (Hz)')
# ax2.set_ylabel('Phase (degrees)')
# ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
#
# plt.tight_layout()
# plt.show()
# ****************************************************************************************
