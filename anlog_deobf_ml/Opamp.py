## *************************************************************************************
import PySpice
import warnings
import PySpice.Logging.Logging as Logging
from PySpice.Doc.ExampleTools import find_libraries

logger = Logging.setup_logging()

from PySpice.Spice.Library import SpiceLibrary
from PySpice.Spice.Netlist import Circuit, SubCircuit, SubCircuitFactory

libraries_path = find_libraries()
spice_library = SpiceLibrary(libraries_path)
from PySpice.Unit import *

# ***************************************************************************************
# OPAMP INTERNAL CIRCUIT USING MOSFET
# ****************************************************************************************


class OpAmp(SubCircuit):
    NAME = 'Operational amplifier'
    NODES = ('non-inv', 'inv', 'vp', 'vn', 'output')

    def __init__(self, name, pv=0, tol=0, mos_model='level3', num_unk=1, ln1=4.0, wd1=25.0, ln2=4.0, wd2=100.0):
        SubCircuit.__init__(self, name, *self.NODES)
        self.mos_id = 0

        if mos_model == 'level1':
            self.model('NMOS', 'NMOS', level=1)
            self.model('PMOS', 'PMOS', level=1)

        elif mos_model == 'level3':
            self.model('NMOS', 'NMOS', level=3, VTO=.79,
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
                                    TH11ETA = 0.046,
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

        # print(f' pv: {pv}, tol:{tol}, unknowns:{num_unk}, wd2 : {wd2}')

        self.R(1, 1, 'vn', 110@u_kOhm)
        # M <name> <drain node> <gate node> <source node> <bulk/substrate node

        if num_unk == 0:
            self.MOSFET(1, 1, 1, 99, 99, model='PMOS', l=4 * 1E-6, w=35 * 1E-6)
        else:
            self.MOSFET('k' + str(self.mos_id + 1), 1, 1, 99, 99, model='PMOS', l=self.per_tol(ln1, tol) * 1E-6,
                        w=self.per_tol(wd1, tol) * 1E-6)
        self.R(2, 99, 'vp', 1@u_Ohm)
        self.R(3, 1, 'vp', 100@u_MOhm)
        self.MOSFET(self.mos_id+2, 3, 1, 98, 98, model='PMOS', l=4 * 1E-6, w=35 * 1E-6)
        self.R(4, 98, 'vp', 1@u_Ohm)
        self.R(5, 3, 'vp', 100@u_MOhm)
        self.MOSFET(self.mos_id+3, 'output', 1, 97, 97,  model='PMOS', l=4 * 1E-6, w=100 * 1E-6)
        self.R(6, 97, 'vp', 1@u_Ohm)
        self.R(7, 'output', 'vp', 100@u_MOhm)
        self.MOSFET(self.mos_id+4, 4, 'inv', 96, 96, model='PMOS', l=4 * 1E-6, w=60 * 1E-6)
        self.R(8, 3, 96, 1@u_Ohm)
        self.R(9, 4, 96, 100@u_MOhm)
        self.MOSFET(self.mos_id+5, 5, 'non-inv', 95, 95, model='PMOS', l=4 * 1E-6, w=60 * 1E-6)
        self.R(10, 95, 3, 1@u_Ohm)
        self.R(11, 5, 95, 100@u_MOhm)
        self.C(1, 5, 16, 1.27@u_pF)
        self.R(12, 5, 16, 100@u_MOhm)
        self.R(13, 16, 6, 1@u_Ohm)
        self.R('L', 6, 'output', 8.75@u_kOhm)
        self.MOSFET(self.mos_id+6, 4, 4, 94, 94, model='NMOS',  l=4 * 1E-6, w=27.5 * 1E-6)
        self.R(14, 94, 'vn', 1@u_Ohm)
        self.R(15, 4, 94, 100@u_MOhm)
        self.MOSFET(self.mos_id+7, 5, 4, 93, 93, model='NMOS', l=self.per_tol(4, tol) * 1E-6, w=self.per_tol(27.5, tol) * 1E-6)
        # self.MOSFET(7, 5, 4, 93, 93, model='NMOS',   l=4 * 1E-6, w=27.5 * 1E-6)
        self.R(16, 93, 'vn', 1@u_Ohm)
        self.R(17, 5, 93, 100@u_MOhm)
        # self.MOSFET(8, 'output', 5, 92, 92, model='NMOS',  l=4 * 1E-6, w=100 * 1E-6)
        if num_unk == 4:
            self.MOSFET('k' + str(self.mos_id+8), 'output', 5, 92, 92, model='NMOS', l=self.per_tol(ln2, tol) * 1E-6,
                        w=self.per_tol(wd2, tol) * 1E-6)

        else:
            wd2 = 100.0
            self.MOSFET(self.mos_id + 8, 'output', 5, 92, 92, model='NMOS', l=self.per_tol(ln2, tol) * 1E-6,
                        w=self.per_tol(wd2, tol) * 1E-6)
        self.R(18, 92, 'vn', 1@u_Ohm)
        self.R(19, 'output', 92, 100@u_MOhm)

    def per_tol(self, x, tole):
        pv_tol = tole
        tor = x + x * (pv_tol / 100)
        # print(f'tolerance: {tor}')
        return tor