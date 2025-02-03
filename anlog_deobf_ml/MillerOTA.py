## *************************************************************************************
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
set_train_path = dataset_config.train_ota_bpf_path_4
# ***************************************************************************************
# OPAMP INTERNAL CIRCUIT USING MOSFET
# ****************************************************************************************
font_size = 10
rc = {"text.usetex": True, "font.family": "serif", "font.weight": "bold", "axes.labelweight": "bold",
          "font.serif": ["Palatino"], "xtick.labelsize": font_size, 'figure.figsize': (5, 4),
          "ytick.labelsize": font_size, 'axes.grid': True, 'axes.facecolor': 'white',
          'grid.linestyle': '--', 'grid.linewidth': 1, 'lines.linewidth': 2.5, "axes.linewidth": 2.5,
          'axes.axisbelow': True}
plt.rcParams.update(rc)
# ****************************************************************************************

class MillerOTA(SubCircuit):
    NAME = 'Miller OTA'
    NODES = ('non_inv', 'inv', 'vdd', 'vss', 'out')

    def __init__(self, name, pv=0, tol=0, mos_model='level3', num_unk=0, ln1=4.0, wd1=25.0, ln2=4.0, wd2=4.934797e+00):
        SubCircuit.__init__(self, name, *self.NODES)
        self.mos_id = 0

        self.w1 = 2.809467e+00
        self.l1 = 2.460902e-01
        self.w3 = 3.928397e+00
        self.l3 = 1.800000e-01
        self.w5 = wd2
        # self.w5 = 50.934797e+00
        self.l5 = 3.077060e-01
        self.w6 = 7.267977e+00
        self.l6 = 4.918375e-01
        self.w7 = 4.566986e+00
        self.l7 = 2.195169e-01
        self.Ib = 7.766726e+01
        # self.Ib = 9.766726e+01
        self.Cc = 8.488920e+00

        if mos_model == 'level1':
            self.model('NMOS', 'NMOS', level=1)
                       # VTO=0.70, KP=110E-6, GAMMA=0.4, LAMBDA=0.04,
                       # PHI=0.7, MJ=0.5, MJSW=0.38, CGBO=700E-12, CGSO=220E-12, CGDO=220E-12,
                       # CJ=770E-6, CJSW=380E-12, LD=0.016E-6, TOX=14E-9)
            self.model('PMOS', 'PMOS', level=1)
                       # VTO=-0.70, KP=50E-6, GAMMA=0.57, LAMBDA=0.05,
                       # PHI=0.8, MJ=0.5, MJSW=0.35, CGBO=700E-12, CGSO=220E-12, CGDO=220E-12,
                       # CJ=560E-6, CJSW=350E-12, LD=0.014E-6, TOX=14E-9)

        elif mos_model == 'level3':
            self.model('NMOS', 'NMOS', level=3,
                                    VTO=.79,
                                    GAMMA=.38,
                                    PHI=.53,
                                    RD=63,
                                    IS=1E-16,
                                    PB=.8,
                                    CGSO=1.973E-10,
                                    CGDO=1.973E-10,
                                    RSH=45,
                                    CJ=0.00029,
                                    MJ = .486,
                                    CJSW = 3.3E-10,
                                    MJSW = .33,
                                    JS = 0.0001,
                                    TOX = 2.5E-08,
                                    NSUB = 8.7E+15,
                                    NFS = 8.2E+11,
                                    TPG = 1,
                                    XJ = 1E-07,
                                    LD = 7E-08,
                                    UO = 577,
                                    VMAX = 150000,
                                    FC = .5,
                                    DELTA = .3551,
                                    THETA = 0.046,
                                    ETA = .16,
                                    KAPPA = 0.05)
            self.model('PMOS', 'PMOS', level=3,
                        VTO = -8.40000000E-01,
                        GAMMA = .53,
                        PHI = .58,
                        RD = 94,
                        RS = 94,
                        IS = 1E-16,
                        PB = .8,
                        CGSO = 3.284E-10,
                        CGDO = 3.284E-10,
                        RSH = 100,
                        CJ = 0.00041,
                        MJ = .54,
                        CJSW = 3.4E-10,
                        MJSW = .3,
                        JS = 0.0001,
                        TOX = 2.5E-08,
                        NSUB = 1.75E+16,
                        NFS = 8.4E+11,
                        TPG = 1,
                        XJ = 0,
                        LD = 6E-08,
                        UO = 205,
                        VMAX = 500000,
                        FC = .5,
                        DELTA = .4598,
                        THETA = .14,
                        ETA = .17,
                        KAPPA = 10
            )

        if pv == 0:
            tol = 0
        elif pv == 1:
            tol = tol
        # M <name> <drain node> <gate node> <source node> <bulk/substrate node

        self.MOSFET(1, 2, 'inv', 3, 3, model='NMOS', l=self.l1 * 1E-6, w=self.w1 * 1E-6)
        self.MOSFET(self.mos_id+2, 5, 'non_inv', 3, 3, model='NMOS', l=self.l1 * 1E-6, w=self.w1 * 1E-6)
        self.MOSFET(self.mos_id+3, 2, 2, 'vdd', 'vdd', model='PMOS', l=self.l3 * 1E-6, w=self.w3 * 1E-6)
        self.MOSFET(self.mos_id+4, 5, 2, 'vdd', 'vdd', model='PMOS', l=self.l3 * 1E-6, w=self.w3 * 1E-6)
        self.MOSFET(self.mos_id+5, 3, 4, 'vss', 'vss', model='NMOS', l=self.l5 * 1E-6, w=self.w5 * 1E-6)
        self.MOSFET(self.mos_id+6, 'out', 5, 'vdd', 'vdd', model='PMOS', l=self.l6 * 1E-6, w=self.w6 * 1E-6)
        self.MOSFET(self.mos_id+7, 'out', 4, 'vss', 'vss', model='NMOS', l=self.l7 * 1E-6, w=self.w7 * 1E-6)
        if num_unk == 0:
            self.MOSFET(self.mos_id + 8, 4, 4, 'vss', 'vss', model='NMOS', l=self.l5 * 1E-6, w=self.w5 * 1E-6)
        else:
            self.MOSFET('k' + str(num_unk) + str(self.mos_id+8), 4, 4, 'vss', 'vss', model='NMOS',
                        l=self.l5 * 1E-6, w=self.w5 * 1E-6)

        if num_unk == 4:
            self.MOSFET('k' + str(self.mos_id+8), 4, 4, 'vss', 'vss', model='NMOS', l=self.per_tol(ln2, tol) * 1E-6,
                        w=self.per_tol(wd2, tol) * 1E-6)
        else:
            self.MOSFET(self.mos_id + 8, 4, 4, 'vss', 'vss', model='NMOS', l=self.l5 * 1E-6, w=self.w5 * 1E-6)

        self.I('bias', 'vdd', 4, self.Ib @ u_uA)
        self.C('c', 5, 'out', self.Cc @ u_pF)
        self.C('l', 'out', self.gnd, 3 @ u_pF)

    def per_tol(self, x, tole):
        pv_tol = tole
        tor = x + x * (pv_tol / 100)
        # print(f'tolerance: {tor}')
        return tor
