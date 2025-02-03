################################################################
# STANDARD DECLARATIONS
import argparse
import math
import os
import random

import numpy as np
import pandas as pd

import PySpice.Logging.Logging as Logging
from PySpice.Doc.ExampleTools import find_libraries
logger = Logging.setup_logging()

from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory

libraries_path = find_libraries()
spice_library = SpiceLibrary(libraries_path)
from PySpice.Unit import *

from Opamp import OpAmp
import path_config


#############################################################################
# OSCILLATORS -> 1. WEIN-BRIDGE 2. RC PHASE SHIFT
#############################################################################

class WeinBridgeOscr(SubCircuit):
    NAME = 'Wein Bridge Oscillator'
    NODES = ('vp', 'vn', 'out')

    def __init__(self, name, pv=0, tol=0, r=10, c=10, num_unk=1, ln1=4.0, wd1=25.0, ln2=4.0, wd2=100.0):
        SubCircuit.__init__(self, name, *self.NODES)

        self.V('ref', 'ref', self.gnd, 0.833 @ u_V)
        self.subcircuit(OpAmp(name='OpAmp1', pv=pv, tol=tol, mos_model='level1', num_unk=num_unk, ln1=ln1, wd1=wd1, ln2=ln2, wd2=wd2))
        self.X('op1', 'OpAmp1', 'non_inv', 'inv', 'vp', 'vn', 'out')

        if num_unk == 2 or num_unk == 3 or num_unk == 4:
            self.R('k2', 'non_inv', 'ref', r @ u_kOhm)
        elif num_unk == 1:
            self.R(21, 'non_inv', 'ref', r @ u_kOhm)

        if num_unk == 3 or num_unk == 4:
            self.C('k3', 'non_inv', self.gnd, c @ u_nF)
        else:
            self.C(21, 'non_inv', self.gnd, c @ u_nF)
        self.R(22, 'out', 1, r @ u_kOhm)
        self.C(22, 1, 'non_inv', c @ u_nF)
        self.R(23, 'out', 'inv', 20 @ u_kOhm)
        self.R(24, 'inv', self.gnd, 10 @ u_kOhm)

        # c1 = self['C1'].capacitance
        # r1 = self['R1'].resistance
        # f_c = self.cal_cutoff()

    def cal_cutoff(self):
        F_c = 1 / (2 * math.pi * (self['C22'].capacitance * self['R22'].resistance))
        # print('cutoff_frequency = {0} Hz'.format(F_c))
        return F_c


##******************** RC Phase shift Oscillator ********************************************
class RCPhaseOscr(SubCircuit):
    NAME = 'RC Phase-shift Oscillator'
    NODES = ('vp', 'vn', 'out')

    def __init__(self, name, pv=0, tol=0, r=10, c=10, num_unk=1, ln1=4.0, wd1=25.0, ln2=4.0, wd2=100.0):
        SubCircuit.__init__(self, name, *self.NODES)

        self.V('ref', 'ref', self.gnd, 2.5 @ u_V)
        self.subcircuit(OpAmp(name='OpAmp1', pv=pv, tol=tol, mos_model='level1',
                              num_unk=num_unk, ln1=ln1, wd1=wd1, ln2=ln2, wd2=wd2))
        self.X('op1', 'OpAmp1', 'ref', 'inv', 'vp', 'vn', 'out')

        if num_unk == 2 or num_unk == 3 or num_unk == 4:
            self.R('k2', 1, 2, r @ u_kOhm)
        elif num_unk == 1:
            self.R(21, 1, 2, r @ u_kOhm)

        if num_unk == 3 or num_unk == 4:
            self.C('k3', 2, self.gnd, c @ u_nF)
        else:
            self.C(21, 2, self.gnd, c @ u_nF)
        self.R(22, 2, 3, r @ u_kOhm)
        self.C(22, 3, self.gnd, c @ u_nF)
        self.R(23, 3, 'out', r @ u_kOhm)
        self.C(23, 'out', self.gnd, c @ u_nF)
        self.R(24, 'out', 'inv', 55.2 @ u_kOhm)       # RG
        self.R(25, 'inv', self.gnd, 1.5 @ u_MOhm)    # RF

        # c1 = self['C1'].capacitance
        # r1 = self['R1'].resistance
        # f_c = self.cal_cutoff()

    def cal_cutoff(self):
        F_c = 1 / (2 * math.pi * (self['C22'].capacitance * self['R22'].resistance))
        # print('cutoff_frequency = {0} Hz'.format(F_c))
        return F_c


#############################################################################
# FILTERS -> 1. LPF (orders -> 1, 2)  2. BPF (order  -> 4)
#############################################################################
class Lpf1stOrder(SubCircuit):
    NAME = 'First order Butterworth LPF'
    NODES = ('v_in', 'vp', 'vn', 'out')

    def __init__(self, name, pv=0, tol=0, r=10, c=1, num_unk=1, ln1=4.0, wd1=25.0, ln2=4.0, wd2=100.0):
        SubCircuit.__init__(self, name, *self.NODES)

        self.subcircuit(OpAmp(name='OpAmp1', pv=pv, tol=tol, mos_model='level1',
                              num_unk=num_unk, ln1=ln1, wd1=wd1, ln2=ln2, wd2=wd2))
        self.X('op1', 'OpAmp1', 'non_inv', 'inv', 'vp', 'vn', 'out')

        if num_unk == 2 or num_unk == 3 or num_unk == 4:
            self.R('k2', 'v_in', 'non_inv', r @u_kOhm)
        else:
            self.R(22, 'v_in', 'non_inv', r @ u_kOhm)
        self.R('s', 'inv', 'out', 0 @ u_Ohm)
        if num_unk == 3 or num_unk == 4:
            self.C('k3', 'non_inv', self.gnd, c @u_uF)
        else:
            self.C(22, 'non_inv', self.gnd, c @ u_uF)
        self.R(23, 'inv', self.gnd, 1 @ u_kOhm)
        self.R(24, 'inv', 'out', 1 @ u_kOhm)

    def cal_cutoff(self, num_unk):
        if num_unk == 1:
            F_c = 1 / (2 * math.pi * (self['C22'].capacitance * self['R22'].resistance))
        elif num_unk == 2:
            F_c = 1 / (2 * math.pi * (self['C22'].capacitance * self['Rk2'].resistance))
        elif num_unk == 3 or num_unk == 4:
            F_c = 1 / (2 * math.pi * (self['Ck3'].capacitance * self['Rk2'].resistance))
        # print('cutoff_frequency = {0} Hz'.format(F_c))
        return F_c


class Lpf2ndOrder(SubCircuit):
    NAME = 'Second order Sallen-key Butterworth LPF'
    NODES = ('v_in', 'vp', 'vn', 'out')

    def __init__(self, name, pv=0, tol=0, r=10, c=1, num_unk=1, ln1=4.0, wd1=25.0, ln2=4.0, wd2=100.0):
        SubCircuit.__init__(self, name, *self.NODES)

        self.subcircuit(OpAmp(name='OpAmp1', pv=pv, tol=tol, mos_model='level1',
                              num_unk=num_unk, ln1=ln1, wd1=wd1, ln2=ln2, wd2=wd2))
        self.X('op1', 'OpAmp1', 'non_inv', 'inv', 'vp', 'vn', 'out')

        if num_unk == 2 or num_unk == 3 or num_unk == 4:
            self.R('k2', 'v_in', 1, r @ u_kOhm)
        else:
            self.R(21, 'v_in', 1, r @ u_kOhm)
        self.R(22, 1, 'non_inv', r @ u_kOhm)
        if num_unk == 3 or num_unk == 4:
            self.C('k3', 'non_inv', self.gnd, c @ u_uF)
        else:
            self.C(21, 'non_inv', self.gnd, c @ u_uF)

        self.C(22, 1, 'out', c @ u_uF)
        self.R(23, 'inv', self.gnd, 4.7 @ u_kOhm)
        self.R(24, 'inv', 'out', 10 @ u_kOhm)

    def cal_cutoff(self, num_unk):
        if num_unk == 0 or num_unk == 1:
            F_c = 1 / (2 * math.pi * (self['C21'].capacitance * self['R21'].resistance))
        elif num_unk == 2:
            F_c = 1 / (2 * math.pi * (self['C21'].capacitance * self['Rk2'].resistance))
        elif num_unk == 3 or num_unk == 4:
            F_c = 1 / (2 * math.pi * (self['Ck3'].capacitance * self['Rk2'].resistance))
        # print('cutoff_frequency = {0} Hz'.format(F_c))
        return F_c


##******************** Band Pass Filter ********************************************
class Bpf4thOrder(SubCircuit):
    NAME = 'Forth order Sallen-key Butterworth BPF'
    NODES = ('v_in', 'vp', 'vn', 'out')

    def __init__(self, name, pv=0, tol=0, r=10, c=1, num_unk=1, ln1=4.0, wd1=25.0, ln2=4.0, wd2=100.0):
        SubCircuit.__init__(self, name, *self.NODES)

        self.C(41, 'v_in', 1, 0.1 @ u_uF)
        self.C(42, 1, 'non_inv_1', 0.1 @ u_uF)
        self.subcircuit(OpAmp(name='OpAmp1', pv=pv, tol=tol, mos_model='level1',
                              num_unk=num_unk, ln1=ln1, wd1=wd1, ln2=ln2, wd2=wd2))
        self.X('op1', 'OpAmp1', 'non_inv_1', 'inv_1', 'vp', 'vn', 'out_1')

        if num_unk == 2 or num_unk == 3 or num_unk == 4:
            self.R('k2', 'non_inv_1', self.gnd, r @ u_kOhm)
        else:
            self.R(41, 'non_inv_1', self.gnd, r @ u_kOhm)

        self.R(42, 1, 'out_1', 10 @ u_kOhm)
        self.R(43, 'inv_1', self.gnd, 1 @ u_kOhm)
        self.R(44, 'inv_1', 'out_1', 1 @ u_kOhm)

        if num_unk == 3 or num_unk == 4:
            self.R('k3', 'out_1', 2, r @ u_kOhm)
        else:
            self.R(45, 'out_1', 2, r @ u_uF)

        self.R(46, 2, 'non_inv_2', r @ u_kOhm)
        self.subcircuit(OpAmp(name='OpAmp2', pv=pv, tol=tol, mos_model='level1',
                              num_unk=num_unk, ln1=ln1, wd1=wd1, ln2=ln2, wd2=wd2))
        self.X('op2', 'OpAmp2', 'non_inv_2', 'inv_2', 'vp', 'vn', 'out')
        self.C(43, 'non_inv_2', self.gnd, 10 @ u_nF)
        self.R(47, 'inv_2', self.gnd, 1 @ u_kOhm)
        self.R(48, 'inv_2', 'out', 1 @ u_kOhm)
        self.C(44, 2, 'out', 10 @ u_nF)

    def cal_cutoff(self, num_unk):
        if num_unk == 1:
            F_c = 1 / (2 * math.pi * (self['C41'].capacitance * self['R41'].resistance))
        elif num_unk == 2:
            F_c = 1 / (2 * math.pi * (self['C41'].capacitance * self['Rk2'].resistance))
        elif num_unk == 3 or num_unk == 4:
            F_c = 1 / (2 * math.pi * (self['C41'].capacitance * self['Rk2'].resistance))
        # print('cutoff_frequency = {0} Hz'.format(F_c))
        return F_c


#######################################################################
# Voltage Controlled Oscillator
#######################################################################
class VCO(SubCircuit):
    NAME = 'Voltage controlled Oscillator'
    NODES = ('v_in', 'vp', 'vn', 'out')

    def __init__(self, name, r=100, c=0.01, pv=0, tol=0, num_unk=0, ln1=4.0, wd1=25.0, ln2=4.0, wd2=100.0):
        SubCircuit.__init__(self, name, *self.NODES)

        self.model('1N4002', 'D',
                   IS=14.11E-11,
                   N=1.984,
                   RS=33.89E-3,
                   IKF=94.81,
                   XTITUN=3,
                   EG=1.110,
                   CJO=51.17E-12,
                   M=.2762,
                   VJ=.3905,
                   FC=0.5,
                   ISR=100.0E-12,
                   NR=2,
                   BV=100.1,
                   IBV=10,
                   TT=4.761E-6)

        self.subcircuit(OpAmp(name='OpAmp1', pv=pv, tol=tol, mos_model='level1',
                              num_unk=num_unk, ln1=ln1, wd1=wd1, ln2=ln2, wd2=wd2))
        self.X('op1', 'OpAmp1', 'non_inv', 'inv', 'vp', 'vn', 'out')

        if num_unk == 2 or num_unk == 3 or num_unk == 4:
            self.R('k2', 1, 'out', r @ u_kOhm)
        else:
            self.R(21, 1, 'out', 100 @ u_kOhm)
        self.R(22, 2, 'v_in', 100 @ u_kOhm)
        self.D('D1', 'out', 2, model='1N4002')
        # circuit.R(5, 2, 'non_inv', 3.3 @ u_kOhm)
        self.R(25, 2, 'non_inv', 0 @ u_kOhm)
        # circuit.R(4, 1, 'inv', 3.3 @ u_kOhm)
        self.R(24, 1, 'inv', 0 @ u_kOhm)
        if num_unk == 3 or num_unk == 4:
            self.C('k3', 1, self.gnd, c @ u_uF)
        else:
            self.C(21, 1, self.gnd, 0.01 @ u_uF)

    def cal_cutoff(self, num_unk):
        if num_unk == 1:
            F_c = 1 / (2 * math.pi * (self['C21'].capacitance * self['R21'].resistance))
        elif num_unk == 2:
            F_c = 1 / (2 * math.pi * (self['C21'].capacitance * self['Rk2'].resistance))
        elif num_unk == 3 or num_unk == 4:
            F_c = 1 / (2 * math.pi * (self['Ck3'].capacitance * self['Rk2'].resistance))
        # print('cutoff_frequency = {0} Hz'.format(F_c))
        return F_c


#######################################################################
# RC CIRCUIT (Random Series and Parallel)
#######################################################################
class SeriesRC(SubCircuit):
    NAME = 'SeriesRC'
    NODES = ('x', 'y')

    def __init__(self, name, ct=1):
        SubCircuit.__init__(self, name, *self.NODES)

        caps = []
        res = u_kOhm(range(1, 10, 1))
        for c in range(5):
            ca = random.uniform(0.001, 0.01)
            caps.append(u_uF(ca))

        r = random.choice(res)
        self.R(ct, 'x', 'y', r)
        c = random.choice(caps)
        self.C(ct, 'y', self.gnd, c)


##******************** Parallel RC Circuit ********************************************
class ParallelRC(SubCircuit):
    NAME = 'ParallelRC'
    NODES = ('x', 'y')

    def __init__(self, name, ct=1):
        SubCircuit.__init__(self, name, *self.NODES)

        caps = []
        res = u_kOhm(range(1, 10, 1))
        for c in range(5):
            ca = random.uniform(0.001, 0.01)
            caps.append(u_uF(ca))

        r = random.choice(res)
        self.R(ct, 'x', 'y', r)
        c = random.choice(caps)
        self.C(ct, 'x', 'y', c)


#######################################################################
# CIRCUIT NETLIST AND SIMULATION
#######################################################################
class DatasetGen:
    def __init__(self, **kwargs) -> None:
        self.dataset_golden = './golden/'
        self.base_path = path_config.train_dataset_base_path
        self.num_of_golden_benches = 400
        self.method = 'transient'
        self.type_of_circuit = ['osc', 'filter', 'bpf']
        self.num_of_res, self.num_of_cap, self.num_of_ind, self.num_of_mos_wd_1, self.num_of_mos_wd_2, self.num_of_inputs = 1, 1, 1, 1, 1, 1
        self.resi = []
        self.cap = []
        self.ind = []
        self.vin = []
        self.v_out = []
        self.ins_pwr = []
        self.mos_ln_1 = []
        self.mos_ln_2 = []
        self.mos_wd_1 = []
        self.mos_wd_2 = []
        self.mean = 0
        self.variance = 0
        self.std_dev = 0
        self.i_vp = 0
        self.i_vn = 0
        self.vdd = 12
        self.avg_pwr = 0
        self.ckt_cnt = 1
        v_input = 0

        tp_of_circuit = args.type_of_circuit
        num_unk = args.no_of_unkn
        data_size = args.size_of_dataset

        if num_unk == 1:
            self.num_of_mos_wd_1 = data_size
            self.num_of_inputs = 1
        elif num_unk == 2:
            size_2 = int(np.round(data_size/2,0))
            # print(f'size: {size_2}')
            self.num_of_mos_wd_1 = size_2
            self.num_of_res = size_2
            # self.num_of_inputs = 2
        elif num_unk == 3:
            size_3 = int(np.round(data_size/3,0))
            self.num_of_mos_wd_1 = size_3
            self.num_of_res = size_3
            self.num_of_cap = size_3
            # self.num_of_inputs = 3
        elif num_unk == 4:
            size_4 = int(np.round(data_size/4, 0))
            self.num_of_mos_wd_1 = size_4
            self.num_of_res = size_4
            self.num_of_cap = size_4
            self.num_of_mos_wd_2 = size_4
            # self.num_of_inputs = 4
        elif num_unk == 0:
            size_4 = int(np.round(data_size/4, 0))
            self.num_of_mos_wd_1 = size_4
            self.num_of_res = size_4
            self.num_of_cap = size_4
            self.num_of_mos_wd_2 = size_4

        self.gen_comp_values(self.num_of_res, self.num_of_cap, self.num_of_ind, self.num_of_mos_wd_1,
                             self.num_of_mos_wd_2, self.num_of_inputs)
        if tp_of_circuit == 'osc':
            v_input = 0
            self.vdd = 12
        elif tp_of_circuit == 'lpf':
            v_input = 5
            self.vdd = 12
        elif tp_of_circuit == 'vco':
            v_input = 2.5
            self.vdd = 5
        self.create_benchmark_circuit(tp_of_circuit, vdd=self.vdd, vss=12, input_vltg=v_input, rdd=700)

    def gen_comp_values(self, num_of_res, num_of_cap, num_of_ind, num_of_wd_1, num_of_wd_2, num_of_inputs):
        for rs in range(num_of_res):
            re = random.uniform(1, 150.5)
            self.resi.append(re)

        for cp in range(num_of_cap):
            ca = random.uniform(0.01, 10.5)
            self.cap.append(ca)

        for id in range(num_of_ind):
            it = random.uniform(0.01, 30.5)
            self.ind.append(it)

        for wd in range(num_of_wd_1):
            wd_1 = random.uniform(60, 270.5)
            self.mos_wd_1.append(wd_1)

        for w in range(num_of_wd_2):
            wd_2 = random.uniform(60, 270.5)
            self.mos_wd_2.append(wd_2)

        for vd in range(num_of_inputs):
            vl = random.uniform(5, 10.5)
            self.vin.append(vl)

        return

    def per_tol(self, x):
        tol = args.pv_tolerance
        pv_tol = random.uniform(0.01, tol)
        # print(f'x before tol: {x}')
        tol = x + x * (pv_tol/100)
        # print(f'x after {pv_tol}% tol: {tol}')
        return tol

    def generate_path_for_benchmarks(self, tp_of_circuit, ckt_class, num_unk, pr_var):
        pr_var = args.process_variation
        pv_tol = args.pv_tolerance
        if pr_var == 0:
            ckt_path = f'{tp_of_circuit}_{ckt_class}_{num_unk}/'
        elif pr_var == 1:
            ckt_path = f'{tp_of_circuit}_{ckt_class}_pv_{pv_tol}_{num_unk}/'

        self.check_dir_exists(self.base_path + ckt_path)
        final_path = os.path.join(self.base_path, ckt_path)
        # print(f'final path: {final_path}')
        return final_path

    def check_dir_exists(self, dirs: str) -> None:
        if not os.path.exists(dirs):
            os.makedirs(dirs)
            print(f'- making dir  ->  {dirs} ')
        return

    def write_bench_dir(self, path, ckt_class, ckt, vout):
        pr_var = args.process_variation
        pv_tol = args.pv_tolerance
        if pr_var == 0:
            f = open(path + f'{ckt_class}_{self.ckt_cnt}.sp', 'w')
        else:
            f = open(path + f'{ckt_class}_pv_{pv_tol}_{self.ckt_cnt}.sp', 'w')
        f.write("VOUT {}\n".format(vout))
        f.write(str(ckt))
        f.close()
        return

    def create_benchmark_circuit(self, tp_of_circuit: str, vdd: int, vss: int, input_vltg: float, rdd: int):
        # df = self.create_df()
        self.method = 'transient'

        ckt_class = None
        num_unk = args.no_of_unkn
        noise_bit = args.noise_input
        pr_var = args.process_variation
        pv_tol = args.pv_tolerance
        osc_tp = args.type_of_osc
        filter_tp = args.type_of_filter
        self.method = args.simulation_method
        length = []
        # print(f'noise: {noise_bit}')
        if tp_of_circuit == 'osc':
            ckt_class = osc_tp
        elif tp_of_circuit == 'filter':
            ckt_class = filter_tp

        path = self.generate_path_for_benchmarks(tp_of_circuit, ckt_class, num_unk, pr_var)

        if tp_of_circuit == 'osc':
            for wd1 in self.mos_wd_1:
                for wd2 in self.mos_wd_2:
                    for res in self.resi:
                        for cap in self.cap:
                            for v in self.vin:
                                circuit = Circuit(osc_tp + '_' + str(self.ckt_cnt))

                                circuit.V('p', 'vdd', circuit.gnd, vdd @ u_V)
                                circuit.R('dd', 'vdd', 'vp', rdd @ u_Ohm)
                                circuit.V('n', 'vn', circuit.gnd, -vss @ u_V)

                                if pr_var == 0:
                                    if osc_tp == 'wb':
                                        circuit.subcircuit(WeinBridgeOscr(name='Osc', r=res, c=cap, wd1=wd1, wd2=wd2,
                                                                          pv=pr_var, tol=pv_tol, num_unk=num_unk))
                                        osc = WeinBridgeOscr(SubCircuit, r=res, c=cap, wd1=wd1, wd2=wd2,
                                                             pv=pr_var, tol=pv_tol, num_unk=num_unk)
                                    elif osc_tp == 'rcps':
                                        circuit.subcircuit(RCPhaseOscr(name='Osc', r=res, c=cap, wd1=wd1, wd2=wd2,
                                                                       pv=pr_var, tol=pv_tol, num_unk=num_unk))
                                        osc = RCPhaseOscr(SubCircuit, r=res, c=cap, wd1=wd1, wd2=wd2,
                                                          pv=pr_var, tol=pv_tol, num_unk=num_unk)
                                elif pr_var == 1:
                                    if osc_tp == 'wb':
                                        circuit.subcircuit(WeinBridgeOscr(name='Osc', r=self.per_tol(res),
                                                                          c=self.per_tol(cap), wd1=wd1, wd2=wd2,
                                                                          pv=pr_var, tol=pv_tol,num_unk=num_unk))
                                        osc = WeinBridgeOscr(SubCircuit, r=self.per_tol(res), c=self.per_tol(cap),
                                                             wd1=wd1, wd2=wd2,
                                                             pv=pr_var, tol=pv_tol,num_unk=num_unk)
                                    elif osc_tp == 'rcps':
                                        circuit.subcircuit(RCPhaseOscr(name='Osc', r=self.per_tol(res),
                                                                       c=self.per_tol(cap), wd1=wd1, wd2=wd2,
                                                                       pv=pr_var, tol=pv_tol,
                                                                       num_unk=num_unk))
                                        osc = RCPhaseOscr(SubCircuit, r=self.per_tol(res), c=self.per_tol(cap),
                                                          wd1=wd1, wd2=wd2,
                                                          pv=pr_var,tol=pv_tol, num_unk=num_unk)

                                circuit.X('osc','Osc', 'vp', 'vn',  'out')
                                # print(circuit)
                                f_c = osc.cal_cutoff()
                                # print(f' cut off frequency : {f_c} Hz')

                                Vout = self.simulate(circuit, self.method, 'out')

                                F_osc = '{0:.6e}'.format(f_c)
                                self.v_out = np.append(Vout, F_osc)
                                self.v_out = self.v_out.astype(float)
                                # print("VOUT: {}".format(self.v_out))

                                self.write_bench_dir(path, ckt_class, circuit, self.v_out)

                                self.ckt_cnt += 1

        elif tp_of_circuit == 'filter':
            for wd1 in self.mos_wd_1:
                for wd2 in self.mos_wd_2:
                    for res in self.resi:
                        for cap in self.cap:
                            for v in self.vin:
                                circuit = Circuit(filter_tp + '_' + str(self.ckt_cnt))
                                circuit.V('p', 'vdd', circuit.gnd, vdd @ u_V)
                                circuit.R('dd', 'vdd', 'vp', rdd @ u_Ohm)
                                circuit.V('n', 'vn', circuit.gnd, -vss @ u_V)
                                circuit.SinusoidalVoltageSource('in', 'v_in', circuit.gnd, amplitude=input_vltg @ u_V,
                                                                frequency=1 @ u_kHz)

                                if pr_var == 0:
                                    if filter_tp == 'lpf_1':
                                        circuit.subcircuit(
                                            Lpf1stOrder(name='lpf_1', r=res, c=cap, wd1=wd1, wd2=wd2,
                                                        pv=pr_var, tol=pv_tol, num_unk=num_unk))
                                        filt = Lpf1stOrder(SubCircuit, r=res, c=cap, wd1=wd1, wd2=wd2,
                                                           pv=pr_var, tol=pv_tol, num_unk=num_unk)
                                    elif filter_tp == 'lpf_2':
                                        circuit.subcircuit(Lpf2ndOrder(name='lpf_2', r=res, c=cap, wd1=wd1, wd2=wd2,
                                                                       pv=pr_var, tol=pv_tol, num_unk=num_unk))
                                        filt = Lpf2ndOrder(SubCircuit, r=res, c=cap, wd1=wd1, wd2=wd2,
                                                          pv=pr_var, tol=pv_tol, num_unk=num_unk)
                                    elif filter_tp == 'bpf_4':
                                        circuit.subcircuit(Bpf4thOrder(name='bpf_4', r=res, c=cap, wd1=wd1, wd2=wd2,
                                                                       pv=pr_var, tol=pv_tol, num_unk=num_unk))
                                        filt = Bpf4thOrder(SubCircuit, r=res, c=cap, wd1=wd1, wd2=wd2,
                                                          pv=pr_var, tol=pv_tol, num_unk=num_unk)
                                elif pr_var == 1:
                                    if filter_tp == 'lpf_1':
                                        circuit.subcircuit(Lpf1stOrder(name='lpf_1', r=self.per_tol(res),
                                                           c=self.per_tol(cap), wd1=wd1, wd2=wd2, pv=pr_var,
                                                           tol=pv_tol, num_unk=num_unk))
                                        filt = Lpf1stOrder(SubCircuit, r=self.per_tol(res), c=self.per_tol(cap),
                                                           wd1=wd1, wd2=wd2,
                                                           pv=pr_var, tol=pv_tol, num_unk=num_unk)
                                    elif filter_tp == 'lpf_2':
                                        circuit.subcircuit(Lpf2ndOrder(name='lpf_2', r=self.per_tol(res),
                                                           c=self.per_tol(cap), wd1=wd1, wd2=wd2,
                                                           pv=pr_var, tol=pv_tol,
                                                           num_unk=num_unk))
                                        filt = Lpf2ndOrder(SubCircuit, r=self.per_tol(res), c=self.per_tol(cap),
                                                           wd1=wd1, wd2=wd2,
                                                           pv=pr_var, tol=pv_tol, num_unk=num_unk)
                                    elif filter_tp == 'bpf_4':
                                        circuit.subcircuit(Bpf4thOrder(name='bpf_4', r=self.per_tol(res),
                                                           c=self.per_tol(cap), wd1=wd1, wd2=wd2,
                                                           pv=pr_var, tol=pv_tol,
                                                           num_unk=num_unk))
                                        filt = Bpf4thOrder(SubCircuit, r=self.per_tol(res), c=self.per_tol(cap),
                                                           wd1=wd1, wd2=wd2,
                                                           pv=pr_var, tol=pv_tol, num_unk=num_unk)

                                circuit.X('filter', filter_tp, 'v_in', 'vp', 'vn','out')
                                print(circuit)
                                f_c = filt.cal_cutoff(num_unk=num_unk)
                                print(f' cut off frequency : {f_c} Hz')

                                Vout = self.simulate(circuit, self.method, 'out')
                                # Vout = self.simulate(circuit, 'ac', 'out')

                                F_osc = '{0:.6e}'.format(f_c)
                                self.v_out = np.append(Vout, F_osc)
                                self.v_out = self.v_out.astype(float)
                                print("VOUT: {}".format(self.v_out))

                                self.write_bench_dir(path, ckt_class, circuit, self.v_out)
                                self.ckt_cnt += 1

        elif tp_of_circuit == 'vco':
            for wd1 in self.mos_wd_1:
                for wd2 in self.mos_wd_2:
                    for res in self.resi:
                        for cap in self.cap:
                            for v in self.vin:
                                circuit = Circuit(filter_tp + '_' + str(self.ckt_cnt))
                                circuit.V('p', 'vdd', circuit.gnd, vdd @ u_V)
                                circuit.R('dd', 'vdd', 'vp', rdd @ u_Ohm)
                                circuit.V('n', 'vn', circuit.gnd, -vss @ u_V)
                                circuit.SinusoidalVoltageSource('in', 'v_in', circuit.gnd, amplitude=input_vltg @ u_V,
                                                                frequency=1 @ u_kHz)

                                if pr_var == 0:
                                    circuit.subcircuit(VCO(name='vco', r=res, c=cap, wd1=wd1, wd2=wd2,
                                                        pv=pr_var, tol=pv_tol, num_unk=num_unk))

                                    vco = VCO(SubCircuit, r=res, c=cap, wd1=wd1, wd2=wd2,
                                                           pv=pr_var, tol=pv_tol, num_unk=num_unk)

                                elif pr_var == 1:
                                    circuit.subcircuit(Lpf1stOrder(name='vco', r=self.per_tol(res),
                                                                   c=self.per_tol(cap), wd1=wd1, wd2=wd2, pv=pr_var,
                                                                   tol=pv_tol, num_unk=num_unk))
                                    vco = VCO(SubCircuit, r=self.per_tol(res), c=self.per_tol(cap),
                                                           wd1=wd1, wd2=wd2,
                                                           pv=pr_var, tol=pv_tol, num_unk=num_unk)

                                    circuit.X('filter', filter_tp, 'v_in', 'vp', 'vn', 'out')
                                    print(circuit)
                                    f_c = vco.cal_cutoff(num_unk=num_unk)
                                    print(f' cut off frequency : {f_c} Hz')

                                    Vout = self.simulate(circuit, self.method, 'out')

                                    F_osc = '{0:.6e}'.format(f_c)
                                    self.v_out = np.append(Vout, F_osc)
                                    self.v_out = self.v_out.astype(float)
                                    print("VOUT: {}".format(self.v_out))

                                    self.write_bench_dir(path, ckt_class, circuit, self.v_out)
                                    self.ckt_cnt += 1

        # self.save_csv('golden_{}'.format(tp_of_circuit), df)
        # print(df)
        # print(self.ckt_cnt)
        return

    def simulate(self, circuit, method: str, param: str):
        tp_of_circuit = args.type_of_circuit
        v_out = []
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        if method == 'transient':
            if tp_of_circuit == 'vco':
                ic = self.vdd @ u_V
                simulator.initial_condition(out=ic)
            else:
                ic = 0 @ u_V
                simulator.initial_condition(out=ic)
            analysis = simulator.transient(step_time=0.1, end_time=.01)
        elif method == 'operating_point':
            analysis = simulator.operating_point()
        elif method == 'ac':
            analysis = simulator.ac(start_frequency=1 @ u_Hz, stop_frequency=1 @ u_MHz, number_of_points=5,
                                    variation='dec')
        else:
            print('invalid simulation method')

        analysis_1 = simulator.operating_point()
        vout = np.array(analysis[param])

        if method == 'ac':
            v1 = [vout[x] for x in range(0, len(vout), 8)]
            for i in range(len(v1)):
                v1[i] = float('{:.6f}'.format(abs(v1[i])))
            v_out = v1
        elif method == 'transient':
            v1 = vout[0]
            v1 = float("{:.6g}".format(v1))
            v_out = np.append(v_out, v1)

            self.mean = np.mean(vout)
            self.variance = np.var(vout)
            self.std_dev = np.std(vout)

            for v in range(len(vout)):
                self.ins_pwr.append(vout[v] ** 2)
            self.avg_pwr = sum(self.ins_pwr) / len(vout)

            # for resi in (circuit.Rdd, circuit.R1):
            #     resi.plus.add_current_probe(circuit)

            node_vn = analysis_1.branches['vn']
            self.i_vn = float(node_vn)

            node_vp = analysis_1.branches['vp']
            self.i_vp = float(node_vp)

            # print(f'Mean: {self.mean}, variance: {self.variance}, std_dev: {self.std_dev}, avg_pwr: {self.avg_pwr},'
            #       f' i-vp: {self.i_vp}, i_vn: {self.i_vn}')

            # df.loc[len(df)] = self.mean, self.variance, self.std_dev, self.avg_pwr, self.i_vp, self.i_vn, Trojan

            features = (self.mean, self.variance, self.std_dev, self.avg_pwr, self.i_vp, self.i_vn)
            v_out = np.append(v_out, features)

        return v_out

    def create_df(self):
        df_name = pd.DataFrame(columns=['mean', 'variance', 'std_dev', 'avg_pwr', 'i_vdd', 'i_vss', 'Trojan'])
        return df_name

    def save_csv(self, filename, from_file):

        trojan = args.infest_trojan
        noise = args.noise_input
        pv = args.process_variation

        if trojan == 0:
            path = './data/golden/{}'.format(args.type_of_circuit)
        elif trojan == 1:
            path = './data/trojan/{}'.format(args.type_of_circuit)

        if not os.path.isdir(path):
            os.makedirs(path)

        if noise == 0:
            if pv == 0:
                to_file = os.path.join(path, '{}.csv'.format(filename))
            elif pv == 1:
                to_file = os.path.join(path, '{}_with_pv.csv'.format(filename))
        elif noise == 1:
            if pv == 0:
                to_file = os.path.join(path, '{}_with_noise.csv'.format(filename))
                print(to_file)
            elif pv == 1:
                to_file = os.path.join(path, '{}_with_noise_pv.csv'.format(filename))

        from_file.to_csv(to_file)
        return


def main(args):
    obj = DatasetGen(**vars(args))
    # obj.gen_trojan_free_benches(args.ben_dir)


def parse_args():
    args = argparse.ArgumentParser(description='Usage: python3 dataset_gen.py')

    # args.add_argument('ben_dir', help='dataset directory for golden benches')
    # args.add_argument('trj_dir', help='dataset directory for trojan benches')
    args.add_argument('-type', '--type_of_circuit', default='osc', type=str,
                      help='Type of a circuit -> osc, filter')
    args.add_argument('-noise', '--noise_input', default=0, type=int,
                      help='Add Noise in the circuit -> 0: no noise or 1: noise')
    args.add_argument('-pv', '--process_variation', default=0, type=int,
                      help=' Add process variation in the circuit -> 0: no variation, 1: variation')
    args.add_argument('-tol', '--pv_tolerance', default=1, type=int,
                      help=' Add % of tolerance for process variation -> between 0 to 90')
    args.add_argument('-unk', '--no_of_unkn', default=1, type=int,
                      help=' Numbr of Unknown components -> between 1 to 4')
    args.add_argument('-sz', '--size_of_dataset', default=500, type=int,
                      help='Number of benches')
    args.add_argument('-osc', '--type_of_osc', default='wb', type=str,
                      help='Type of Oscillator circuit -> 1: wein bridge, 2: rc phase shift')
    args.add_argument('-filter', '--type_of_filter', default='lpf_1', type=str,
                      help='Type of Filter circuit -> 1: LPF 1st order, 2: LPF 2nd order')
    args.add_argument('-sim', '--simulation_method', default='transient', type=str,
                      help='type of simulation method -> transient, operating_point, ac')
    # args.add_argument('-routs', '--outputs_range', default=(4, 15), type=tuple,
    #                   help='Range of circuit outputs')
    # args.add_argument('-rdepth', '--depth_range', default=(6, 20), type=tuple,
    #                   help='Range of a circuit')


    return args.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)



