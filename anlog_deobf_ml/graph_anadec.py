import os.path
import sys
import re

import casadi
import sympy
import scipy
from casadi import casadi
from scipy import stats

#import GCN_Learn_Analog
import argparse
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
import copy
from collections import defaultdict
from sklearn import preprocessing

from enum import Enum, IntEnum

import warnings


warnings.filterwarnings('ignore')


class cattr(IntEnum):
    CID = 0
    DEVICE = 1
    ISUNK = 2
    TERMINALS = 3
    NAME = 4
    DTYPE = 5


class devtype(IntEnum):
    RES = 1
    CAP = 2
    IND = 3
    PMOS = 4
    NMOS = 5
    VSRC = 6
    ISRC = 7
    VDD = 8
    INPUT = 9
    OUTPUT = 10
    INTER = 11
    GND = 0
    UNKNW = 12
    VOUT = 13
    SW = 14


class ndattr(IntEnum):
    NID = 0
    VVAR = 1
    VTVAR = 2
    VPVAR = 3
    IVAR = 4
    NEIGHBORS = 5
    NDTYPE = 6
    NAME = 7


class ndtype(IntEnum):
    INPUT = 0
    OUTPUT = 1
    INTER = 2
    GND = 3


class srcattr(IntEnum):
    SID = 0
    SRCTYPE = 1
    SRCSIG = 2
    SRCVAL = 3
    SRCTERMINALS = 4


class srctype(IntEnum):
    V = 0
    I = 1


class srcfun(IntEnum):
    DC = 0
    AC = 1
    PULSE = 2


class staticSolverType(IntEnum):
    SCIPY = 0
    CASADI = 1


class dynamicSolverType(IntEnum):
    CASADI = 0
    SYMPY = 1


si_prefixes = {
    'y': 1e-24,  # yocto
    'z': 1e-21,  # zepto
    'a': 1e-18,  # atto
    'f': 1e-15,  # femto
    'p': 1e-12,  # pico
    'n': 1e-9,  # nano
    'u': 1e-6,  # micro
    'm': 1e-3,  # mili
    'c': 1e-2,  # centi
    'd': 1e-1,  # deci
    'k': 1e3,  # kilo
    'Meg': 1e6,  # mega
    'G': 1e9,  # giga
    'T': 1e12,  # tera
    'P': 1e15,  # peta
    'E': 1e18,  # exa
    'Z': 1e21,  # zetta
    'Y': 1e24,  # yotta
}


class symbolsType(IntEnum):
    SYMPY = 0
    CASADI = 1


def value_str2num(value_str):
    # print('value_str: ', value_str)
    m = re.match(r'(^-?\d*[.]?\d*)([a-zA-Z]*)', value_str)
    value = float(m.group(1))
    # print('group1 :', value)

    if len(m.group(2)) != 0:
        # if m.group(2) == 'Meg':
        #     value = value * si_prefixes[m.group(2)] * 1e-3
        # else:
        value = value * si_prefixes[m.group(2)]
    elif len(m.group(2)) == 0:
        print(f'group2: {m.group(2)}')
        # value = value + 1e3

    print('value {} -> {}'.format(value_str, float(value)))
    return value


class Device:
    def __init__(self, name, cid, Vs=None, Vps=None, T=None):
        self.name = name
        self.cid = cid
        self.Vs = Vs
        self.Vps = Vps
        self.T = T
        self.Is = None
        self.lapIs = None
        self.Eqs = None

    def has_auxEquations(self):
        return 0

    def is_source(self):
        return False

    def dciequations(self):
        raise NotImplementedError

    def iequations(self):
        raise NotImplementedError

    def lapliequations(self):
        raise NotImplementedError

    def getvalue(self):
        raise NotImplementedError


class Resistor(Device):
    def __init__(self, name, cid, value, Vs=None, Vps=None, T=None):
        super().__init__(name, cid, Vs, Vps, T)
        self.Rval = value

    def iequations(self):
        assert len(self.Vs) == 2
        if self.Is is None:
            dv = self.Vs[0] - self.Vs[1]
            self.Is = [dv / self.Rval, -dv / self.Rval]
        return self.Is

    def dciequations(self):
        return self.iequations()

    def getvalue(self):
        return self.Rval


class Capacitor(Device):
    def __init__(self, name, cid, value, Vs=None, Vps=None, T=None):
        super().__init__(name, cid, Vs, Vps, T)
        self.Cval = value

    def iequations(self):
        if self.Is is None:
            assert len(self.Vs) == 2 and len(self.Vps) == 2
            dvp = self.Vps[0] - self.Vps[1]
            i = self.Cval * dvp
            self.Is = [i, -i]
        return self.Is

    def dciequations(self):
        if self.Is is None:
            self.Is = [0, 0]
        return self.Is

    def getvalue(self):
        return self.Cval


class Inductor(Device):
    def __init__(self, name, cid, value, Vs=None, Vps=None, ivar=None, ivari=None, T=None):
        super().__init__(name, cid, Vs, Vps, T)
        # assert len(Vs) == 2 and len(Vps) == 2
        self.Lval = value
        self.ivar = ivar

    def has_auxEquations(self):
        return 1

    def iequations(self):
        if self.Is is not None:
            return self.Is
        # assert self.T is not None and len(self.Vs) == 2 and self.ivar is not None
        v = self.Vs[0] - self.Vs[1]
        self.ivar = v * 1/self.Lval
        self.Is = [self.ivar, - self.ivar]
        return self.Is

    def auxEquations(self):
        if self.Eqs is not None:
            return self.Eqs
        assert self.T is not None and len(self.Vs) == 2 and self.ivar is not None
        dv = self.Vs[0] - self.Vs[1]
        self.Eqs = [self.Lval * casadi.jacobian(self.ivar, self.T) - dv]
        return self.Eqs

    def getvalue(self):
        return self.Lval


class VSource(Device):
    def __init__(self, name, cid, stp, value, Vs=None, ivar=None, T=None):
        super().__init__(name, cid, Vs, T)
        self.srctype = stp
        self.value = value
        self.ivar = ivar

    def is_source(self):
        return True

    def has_auxEquations(self):
        return 1

    def iequations(self):
        if self.Is is None:
            self.Is = [self.ivar, -self.ivar]
        assert len(self.Vs) == 2
        return self.Is

    def dciequations(self):
        return self.iequations()

    def auxEquations(self):
        if self.Eqs is not None:
            return self.Eqs
        assert len(self.Vs) == 2
        print(f'vsource vs: {self.Vs}')
        self.Eqs = [self.Vs[1] - self.Vs[0] + self.value]
        return self.Eqs


class ISource(Device):
    def __init__(self, name, cid, stp, value, Vs=None, T=None):
        super().__init__(name, cid, Vs, T)
        self.srctype = stp
        self.value = value

    def is_source(self):
        return True

    def iequations(self):
        if self.Is is not None:
            return self.Is
        assert len(self.Vs) == 2
        self.Is = [self.value, -self.value]
        return self.Is

    def getvalue(self):
        return self.value


class Mosfet1(Device):
    def __init__(self, se, name, modelnm, cid, Vs=None, T=None,  w=float(1e-6), l=float(1e-6), **kwargs):
        super().__init__(name, cid, Vs, T)
        self.se = se
        self.modelnm = modelnm
        self.is_nmos = (modelnm == 'nmos')
        self.KP = 200e-6 if self.is_nmos else 100e-6
        self.VTO = 0.3 if self.is_nmos else -0.3
        self.LAMBDA = 0.5
        self.W = float(w)
        self.L = float(l)
        self.__dict__.update(kwargs)

    def getSym(self, nm):
        return casadi.MX.sym(nm)

    def floor(self, x):
        return casadi.floor(x)

    def dciequations(self):
        if self.Is is not None:
            return self.Is
        assert len(self.Vs) == 4
        print(f'Vs: {self.Vs}')
        vd, vg, vs, vb = self.Vs
        vgs = (vg - vs) if self.is_nmos else (vs - vg)
        print(f'vgs: {vgs}')
        print(f'vgs: {type(vgs)}')
        vto = self.VTO if self.is_nmos else -self.VTO
        vds = vd - vs
        print(f'vds: {vds}')
        ivdson = self.KP * (self.W / self.L) * ((vgs - vto) * vds - vds ** 2 / 2) * (1 + self.LAMBDA * vds)
        print('ivdson:', ivdson)
        ivdsoff = 0
        ivdssat = 1 / 2 * self.KP * (self.W / self.L) * ((vgs - vto) ** 2) * (1 + self.LAMBDA * vds)
        print('ivds_sat:', ivdssat)
        # ivds = self.se.if_else(vgs <= vto, ivdsoff, self.se.if_else(vds <= (vgs - vto), ivdson, ivdssat))
        if vgs <= vto:
            ivds = ivdsoff
        elif vds <= (vgs - vto):
            ivds = ivdson
        else:
            ivds = ivdssat
        print('ivds:', ivds)
        self.Is = [ivds, 0, -ivds, 0]
        return self.Is

    def idsfun(self, vg, vd, vs):
        vgs = (vg - vs) if self.is_nmos else (vs - vg)
        vto = self.VTO if self.is_nmos else -self.VTO
        vds = vd - vs
        ivdson = self.KP * (self.W / self.L) * ((vgs - vto) * vds - vds ** 2 / 2) * (1 + self.LAMBDA * vds)
        ivdsoff = 0
        ivdssat = 1 / 2 * self.KP * (self.W / self.L) * ((vgs - vto) ** 2) * (1 + self.LAMBDA * vds)
        ids = ivdsoff if vgs <= vto else (ivdson if vds <= (vgs - vto) else ivdssat)
        return ids

    def iequations(self):
        if self.Is is None:
            assert len(self.Vs) == 4 and len(self.Vps) == 4
            self.Is = self.dciequations()
        return self.Is


class CasadiSymEngine:
    def __init__(self):
        self.dynamic_solver_type = dynamicSolverType.CASADI

    def if_else(self, cond, i, e):
        return casadi.if_else(cond, i, e)


class AnaCircuit:
    def __init__(self, bench_file=None, **kwarg):

        self.circuit_name = str()

        self.mos_ids = dict()
        self.comp_ids = dict()
        self.src_ids = dict()
        self.idcounter = 0
        self.devnm2id = dict()
        self.nodenm2id = dict()
        self.nodes = dict()
        self.inputs = set()
        self.outputs = set()
        self.internals = set()
        self.unkmap = dict()
        self.devices = dict()
        self.devnum = defaultdict(lambda: 0)

        self.vertices = set()
        self.edges = set()
        self.unk_comp = set()
        self.wires = dict()
        self.features = dict()
        self.feat_vect = dict()
        self.circuit_X = dict()
        self.circuit_Y = dict()

        self.index_list = dict()
        self.comp_index = dict()
        self.queries = dict()
        self.dev_cnt = 0
        self.comp_cnt = 0
        self.ct = 0
        self.unknown_count = 0
        self.num_vertices = 0
        self.num_edges = 0
        self.num_pmos = 0
        self.num_nmos = 0
        self.num_res = 0
        self.num_ind = 0
        self.num_cap = 0
        self.num_diodes = 0
        self.num_IO = 0
        self.vin_val = 0
        self.iin_val = 0
        self.frq_val = 0
        self.vout_val = []
        self.num_of_queries = 1
        self.max_cnt = []
        self.maxcount = 0
        self.T = None
        self.se = CasadiSymEngine
        self.symType = symbolsType.SYMPY

        return

    @staticmethod
    def value_str2num(value_str):
        print('value_str: ', value_str)
        # m = re.match('(\d*\.\d+|\d+)([ ]*)([a-zA-Z]*)', value_str)
        m = re.search(r'(\d*[.]\d*)([a-zA-Z]*)', value_str)
        value = float(m.group(1))
        # print('group1 :', value)

        if len(m.group(2)) != 0:
            value = value * si_prefixes[m.group(2)]

        # print('value {} -> {.7f}'.format(value_str, float(value)))
        return value

    def read_bench(self, bench_file):
        with open(bench_file, 'r') as file:
            for line, contents in enumerate(file):
                org_contents = contents
                contents = contents.lower()
                tokens = re.split('[ \t()]+', org_contents.strip())
                tokens = list(filter(lambda x: x != '', tokens))
                # print(tokens)
                if line == 1:
                    # self.circuit_name = re.sub("[\b\.\d]", ' ', contents)
                    self.circuit_name = contents.replace(".title", "")
                elif contents.startswith('r') or contents.startswith('c') or contents.startswith('m') \
                        or contents.startswith('v') or contents.startswith('i') or contents.startswith('l'):
                    self.parse_device(tokens)
                elif contents.startswith('.') or contents.startswith('x'):
                    pass

        self.gen_features(tokens)

        print("Parsing bench file for : ", self.circuit_name)
        print("\nFeatures: ", self.features)
        print(f'Devices: {self.devices}\n')
        print(f'Nodes: {self.nodes}')
        # print("\nFeature Vector: ", self.feat_vect)

        self.circuit_X.setdefault('Features', []).extend([self.dev_cnt, self.num_IO, self.num_res,
                                                          self.num_cap, self.num_ind, self.num_pmos,
                                                          self.num_nmos, self.unknown_count, self.vin_val, self.frq_val])
        self.circuit_X.setdefault('Features', []).extend(self.vout_val)

        return self.vertices, self.wires, self.features, self.circuit_name, self.circuit_X, self.circuit_Y

    def parse_device(self, tokens):
        comp_name = str(tokens[0].lower())
        # print(comp_name)
        if comp_name in self.devnm2id:
            raise Exception('device name {} already exists\n'.format(comp_name))

        dev = None
        devtp = None
        m_tp = None
        unk_dev = None
        tnids = []
        isunk = False
        is_unk = 0
        ct = 0
        value = val = truevalue = 0
        uk_val = 'k'
        # se = self.se

        if comp_name.startswith('i') or comp_name.startswith('v'):
            print('parsing source ', tokens)

            if tokens[0].lower() == 'vin' or tokens[0].lower() == 'vout' or tokens[0].lower() == 'vinput':
                if tokens[0].lower() == 'vout': #list of output values appended to netlist after simulating the circuit with pyspice
                    for x in range(1, 6):
                        try:
                            self.vout_val.append(float(tokens[x].strip(",[]")))
                        except ValueError:
                            print("error on line", x)
                    self.vout_val = self.vout_val
                    print(f'vout:{self.vout_val}')

                elif tokens[0].lower() == 'vin' or tokens[0].lower() == 'vinput':
                    nd1nm, nd2nm = tokens[1], tokens[2]
                    tnids = [self.getadd_node(nd1nm), self.getadd_node(nd2nm)]
                    devtp = devtype.VSRC
                    ct = self.comp2index(comp_name)
                    if len(tokens) > 4 and tokens[5] == 'AC':
                        src_fn = srcfun.AC
                        src_tp = srctype.V
                        self.vin_val = val = tokens[9] = value_str2num(tokens[9].strip("V"))
                        self.frq_val = f_val = tokens[10] = value_str2num(tokens[10].strip("Hz"))
                        dev = VSource(tokens[0], ct, devtp, val)
                    elif len(tokens) > 4 and tokens[5] == 'PULSE':
                        src_fn = srcfun.PULSE
                        src_tp = srctype.V
                        self.vin_val = val = tokens[7] = value_str2num(tokens[7].strip("V"))
                        self.frq_val = 0
                        dev = VSource(tokens[0], ct, devtp, val)
                    elif len(tokens) < 5:
                        src_fn = srcfun.DC
                        src_tp = srctype.V
                        self.vin_val = val = tokens[3] = value_str2num(tokens[3].strip("V"))
                        self.frq_val = 0
                        dev = VSource(tokens[0], ct, devtp, val)
                    else:
                        src_fn = srcfun.DC
                        src_tp = srctype.V
                        self.vin_val = tokens[3] = value_str2num(tokens[3].strip("V"))
                        self.frq_val = 0
            elif (tokens[0].lower() == 'vdd' or tokens[0].lower() == 'vss' or tokens[0].lower() == 'vp' or
                  tokens[0].lower() == 'vn' or tokens[0].lower() == 'vgg'):
                nd1nm, nd2nm = tokens[1], tokens[2]
                tnids = [self.getadd_node(nd1nm), self.getadd_node(nd2nm)]
                devtp = devtype.VSRC
                src_tp = srctype.V
                ct = self.comp2index(comp_name)
                self.vin_val = val = tokens[3] = value_str2num(tokens[3].strip("V"))
                dev = VSource(tokens[0], ct, devtp, val)
                # self.vin_val = tokens[3].strip("V")

            elif tokens[0][0].lower() == 'i':
                nd1nm, nd2nm = tokens[1], tokens[2]
                tnids = [self.getadd_node(nd1nm), self.getadd_node(nd2nm)]
                devtp = devtype.ISRC
                src_tp = srctype.I
                ct = self.comp2index(comp_name)
                self.iin_val = val = tokens[3] = value_str2num(tokens[3].strip("A"))
                self.vout_val = self.vout_val
                dev = ISource(tokens[0], ct, devtp, val)

            else:
                nd1nm, nd2nm = tokens[1], tokens[2]
                tnids = [self.getadd_node(nd1nm), self.getadd_node(nd2nm)]
                devtp = devtype.VSRC
                ct = self.comp2index(comp_name)
                if len(tokens) > 4 and tokens[5] == 'AC':
                    src_fn = srcfun.AC
                    src_tp = srctype.V
                    self.vin_val = val = tokens[9] = value_str2num(tokens[9].strip("V"))
                    self.frq_val = f_val = tokens[10] = value_str2num(tokens[10].strip("Hz"))
                    dev = VSource(tokens[0], ct, devtp, val)
                elif len(tokens) > 4 and tokens[5] == 'PULSE':
                    src_fn = srcfun.PULSE
                    src_tp = srctype.V
                    self.vin_val = val = tokens[7] = value_str2num(tokens[7].strip("V"))
                    self.frq_val = 0
                    dev = VSource(tokens[0], ct, devtp, val)
                elif len(tokens) < 5:
                    src_fn = srcfun.DC
                    src_tp = srctype.V
                    self.vin_val = val = tokens[3] = value_str2num(tokens[3].strip("V"))
                    self.frq_val = 0
                    dev = VSource(tokens[0], ct, devtp, val)

        elif comp_name.startswith('m'):
            tp = tokens[5].lower()
            ct = self.comp2index(comp_name)
            if tp.startswith('p'):
                devtp = m_tp = devtype.PMOS
                modelnm = 'pmos'
            else:
                devtp = m_tp = devtype.NMOS
                modelnm = 'nmos'
            # L = tokens[6] = value_str2num(tokens[6].strip("l="))
            # W = tokens[7] = value_str2num(tokens[7].strip("w="))
            L = tokens[6] = tokens[6].strip("l=")
            W = tokens[7] = tokens[7].strip("w=")
            # dev = Mosfet1(self.se, comp_name, modelnm, ct, None, None)

            if comp_name[1] == 'k':
                devtp = devtype.UNKNW
                is_unk = 1
                unk_dev = m_tp
            print('parsing mos ', tokens)
            nmd, nmg, nms, nmb = pnames = tokens[1:5]
            ndid, ngid, nsid, nbid = tnids = [self.getadd_node(x) for x in pnames]
            dev = Mosfet1(self.se, comp_name, modelnm, ct, w=W, l=L)

        elif comp_name.startswith('r') or comp_name.startswith('c') or comp_name.startswith('l'):
            nm1 = tokens[1]
            nm2 = tokens[2]
            nid1 = self.getadd_node(nm1)
            nid2 = self.getadd_node(nm2)
            tnids = [nid1, nid2]
            nms = [nm1, nm2]
            if comp_name.startswith('r'):
                devtp = devtype.RES
                ct = self.comp2index(comp_name)
                rval = truevalue = tokens[3] = value_str2num(tokens[3].strip("Ohm"))
                dev = Resistor(comp_name, ct, rval)
                if comp_name[1] == 'k':
                    unk_dev = devtp
                    is_unk = 1
                    rval = truevalue
                print('parsing resistor ', tokens)

            elif comp_name.startswith('c'):
                devtp = devtype.CAP
                ct = self.comp2index(comp_name)
                cval = truevalue = tokens[3] = value_str2num(tokens[3].strip("F"))

                if comp_name[1] == 'k':
                    unk_dev = devtp
                    is_unk = 1
                    cval = truevalue
                dev = Capacitor(comp_name, ct, cval)
                print('parsing capacitor ', tokens)

            elif comp_name.startswith('l'):
                devtp = devtype.IND
                ct = self.comp2index(comp_name)
                lval = truevalue = tokens[3] = value_str2num(tokens[3].strip("H"))
                dev = Inductor(comp_name, ct, lval)
                if comp_name[1] == 'k':
                    unk_dev = devtp
                    is_unk = 1
                    lval = truevalue
                print('parsing inductor ', tokens)

        elif comp_name.startswith('s'):
            nip, nim, nop, nom = pnames = tokens[1:5]
            ndid, ngid, nsid, nbid = tnids = [self.getadd_node(x) for x in pnames]
            devtp = devtype.SW
            ct = self.comp2index(comp_name)
            truevalue = 0


        else:
            print('unrecognized device at line {}'.format(tokens))

        if comp_name.startswith('m'):
            if is_unk == 1:
                # Features =>[rval, cval, lval, MOS(L), MOS(W), 1=>if unknown, Vin, Vout(5 samples)]
                self.feat_vect.setdefault('Features', []).append([0, 0, 0, 0, 0, is_unk, self.vin_val])
                self.queries.setdefault('q_dev', []).append([unk_dev, W])
                self.feat_vect.setdefault('Vout', []).append(self.vout_val)
                self.circuit_Y.setdefault('UN_DEV-TYPE', []).extend([unk_dev])
                self.circuit_Y.setdefault('UN_MOS_LN', []).extend([L])
                self.circuit_Y.setdefault('UN_MOS_WD', []).extend([W])
                self.circuit_Y.setdefault('UN_R_Val', []).extend([0])
                self.circuit_Y.setdefault('UN_C_Val', []).extend([0])
                self.circuit_Y.setdefault('UN_L_Val', []).extend([0])

            else:
                self.feat_vect.setdefault('Features', []).append([0, 0, 0, W, L, is_unk, self.vin_val])
                self.circuit_X.setdefault('MOS_LN', []).extend([L])
                self.circuit_X.setdefault('MOS_WD', []).extend([W])
                self.circuit_X.setdefault('R_Val', []).extend([0])
                self.circuit_X.setdefault('C_Val', []).extend([0])
                self.circuit_X.setdefault('L_Val', []).extend([0])
                self.feat_vect.setdefault('Vout', []).append(self.vout_val)

        elif comp_name.startswith('r') or comp_name.startswith('c') or comp_name.startswith('l'):
            if is_unk == 1:
                # print("vin_val", self.vin_val)
                self.circuit_Y.setdefault('UN_DEV-TYPE', []).extend([unk_dev])
                self.circuit_Y.setdefault('UN_MOS_LN', []).extend([0])
                self.circuit_Y.setdefault('UN_MOS_WD', []).extend([0])
                if comp_name.startswith('r'):
                    self.feat_vect.setdefault('Features', []).append([0, 0, 0, 0, 0, is_unk, self.vin_val])
                    self.queries.setdefault('q_dev', []).append([unk_dev, rval])
                    self.circuit_Y.setdefault('UN_R_Val', []).extend([rval])
                    self.circuit_Y.setdefault('UN_C_Val', []).extend([0])
                    self.circuit_Y.setdefault('UN_L_Val', []).extend([0])
                elif comp_name.startswith('c'):
                    self.feat_vect.setdefault('Features', []).append([0, 0, 0, 0, 0, is_unk, self.vin_val])
                    self.queries.setdefault('q_dev', []).append([unk_dev, cval])
                    self.circuit_Y.setdefault('UN_C_Val', []).extend([cval])
                    self.circuit_Y.setdefault('UN_R_Val', []).extend([0])
                    self.circuit_Y.setdefault('UN_L_Val', []).extend([0])
                elif comp_name.startswith('l'):
                    self.feat_vect.setdefault('Features', []).append([0, 0, 0, 0, 0, is_unk, self.vin_val])
                    self.queries.setdefault('q_dev', []).append([unk_dev, lval])
                    self.circuit_Y.setdefault('UN_L_Val', []).extend([lval])
                    self.circuit_Y.setdefault('UN_C_Val', []).extend([0])
                    self.circuit_Y.setdefault('UN_R_Val', []).extend([0])
                self.feat_vect.setdefault('Vout', []).append(self.vout_val)
            else:
                self.circuit_X.setdefault('MOS_LN', []).extend([0])
                self.circuit_X.setdefault('MOS_WD', []).extend([0])
                if comp_name.startswith('r'):
                    self.feat_vect.setdefault('Features', []).append([rval, 0, 0, 0, 0, is_unk, self.vin_val])
                    self.circuit_X.setdefault('R_Val', []).extend([rval])
                    self.circuit_X.setdefault('C_Val', []).extend([0])
                    self.circuit_X.setdefault('L_Val', []).extend([0])
                elif comp_name.startswith('c'):
                    self.feat_vect.setdefault('Features', []).append([0, cval, 0, 0, 0, is_unk, self.vin_val])
                    self.circuit_X.setdefault('C_Val', []).extend([cval])
                    self.circuit_X.setdefault('R_Val', []).extend([0])
                    self.circuit_X.setdefault('L_Val', []).extend([0])
                elif comp_name.startswith('l'):
                    self.feat_vect.setdefault('Features', []).append([0, 0, lval, 0, 0, is_unk, self.vin_val])
                    self.circuit_X.setdefault('L_Val', []).extend([lval])
                    self.circuit_X.setdefault('C_Val', []).extend([0])
                    self.circuit_X.setdefault('R_Val', []).extend([0])
                self.feat_vect.setdefault('Vout', []).append(self.vout_val)
        elif comp_name.startswith('i') or comp_name.startswith('v'):
            if tokens[0].lower() != 'vout':
                if comp_name.startswith('i'):
                    self.feat_vect.setdefault('Features', []).append([0, 0, 0, 0, 0, is_unk, self.iin_val])
                    self.feat_vect.setdefault('Vout', []).append(self.vout_val)
                else:
                    self.feat_vect.setdefault('Features', []).append([0, 0, 0, 0, 0, is_unk, self.vin_val])
                    self.feat_vect.setdefault('Vout', []).append(self.vout_val)
            elif tokens[0].lower() == 'vout':
                pass
        if tokens[0].lower() != 'vout':
            # Add to features [Component Id, Component Type(R/L/C/PMOS/NMOS)]
            self.feat_vect.setdefault('Component ID', []).append(ct)
            self.feat_vect.setdefault('Component Type', []).append(devtp)
            self.queries.setdefault('Component ID', []).append(ct)

        for i, tnid in enumerate(tnids):
            self.nodes[tnid][ndattr.NEIGHBORS].add((ct, i))

        # assert devtp is not None
        self.devnum[devtp] += 1

        comp = self.devices[ct] = {cattr.CID: ct,
                                   cattr.TERMINALS: tnids,
                                   cattr.NAME: comp_name,
                                   cattr.DEVICE: dev,
                                   cattr.ISUNK: is_unk,
                                   cattr.DTYPE: devtp}

        vertices = self.add_node(tokens)
        # print(f'vertices: {vertices}')
        edges, wires, features = self.add_edges(tokens)
        # print(f'edges: {edges}, wires: {wires}, features: {features}')

        return

    def getadd_node(self, ndnm):
        if ndnm in self.nodenm2id:
            return self.nodenm2id[ndnm]
        else:
            nid = self.idcounter
            self.nodenm2id[ndnm] = nid
            ndt = self.str2ndtype(ndnm)
            self.register_ndtype(nid, ndt)
            self.nodes[nid] = {ndattr.NID: nid,
                               ndattr.NAME: ndnm,
                               ndattr.NEIGHBORS: set(),
                               ndattr.NDTYPE: ndt}
            self.idcounter += 1
            return nid

    def comp2index(self, comp):
        if comp in self.comp_index:
            return self.comp_index[comp]
        else:
            cnt = self.comp_cnt
            self.comp_index[comp] = cnt
            self.comp_cnt += 1
            return cnt

    def str2ndtype(self, nm):
        if nm.startswith('x'):
            return ndtype.INPUT
        elif nm.startswith('y') or nm == 'out':
            return ndtype.OUTPUT
        elif nm == '0':
            return ndtype.GND
        else:
            return ndtype.INTER

    def remove_device(self, cid):
        for i, tnid in enumerate(self.devices[cid][cattr.TERMINALS]):
            self.nodes[tnid][ndattr.NEIGHBORS].remove((cid, i))
        del self.devnm2id[self.devices[cid][cattr.NAME]]
        self.devnum[self.devices[cid][cattr.DTYPE]] -= 1
        del self.devices[cid]

    def register_ndtype(self, nid, ndt):
        if ndt == ndtype.INPUT:
            self.inputs.add(nid)
        elif ndt == ndtype.OUTPUT:
            self.outputs.add(nid)
        elif ndt == ndtype.INTER:
            self.internals.add(nid)
        elif ndt == ndtype.GND:
            self.gndid = nid

    def add_node(self, tokens):
        st = tokens[0] = tokens[0].lower()

        if st.startswith('r'):
            self.vertices.add(tokens[0])
            self.comp_ids.setdefault('Resistor', []).append(st)
            self.num_vertices += 1
            self.num_res += 1
            self.dev_cnt += 1
            if st[1] == 'k':
                self.unknown_count += 1
                self.unk_comp.add(tokens[0])

        elif st.startswith('c'):
            self.vertices.add(tokens[0])
            self.comp_ids.setdefault('Capacitor', []).append(st)
            self.num_vertices += 1
            self.num_cap += 1
            self.dev_cnt += 1
            if st[1] == 'k':
                self.unknown_count += 1
                self.unk_comp.add(tokens[0])

        elif st.startswith('l'):
            self.vertices.add(tokens[0])
            self.comp_ids.setdefault('Inductor', []).append(st)
            self.num_vertices += 1
            self.num_ind += 1
            self.dev_cnt += 1
            if st[1] == 'k':
                self.unknown_count += 1
                self.unk_comp.add(tokens[0])

        elif st.startswith('m'):
            tp = tokens[5].lower()
            if tp.startswith('p'):
                self.vertices.add('p'+tokens[0])
                self.comp_ids.setdefault('PMOS', []).append(st)
                self.num_vertices += 1
                self.num_pmos += 1
                self.dev_cnt += 1
                if st[1] == 'k':
                    self.unknown_count += 1
                    self.unk_comp.add(tokens[0])

            elif tp.startswith('n'):
                self.vertices.add('n'+tokens[0].lower())
                self.comp_ids.setdefault('NMOS', []).append(st)
                self.num_vertices += 1
                self.num_nmos += 1
                self.dev_cnt += 1
                if st[1] == 'k':
                    self.unknown_count += 1
                    self.unk_comp.add(tokens[0])

        elif st.startswith('v') or st.startswith('i'):
            if st != 'vout':
                self.vertices.add(tokens[0].lower())
                self.comp_ids.setdefault('IO', []).append(st)
                self.num_vertices += 1
                self.num_IO += 1
                # self.dev_cnt += 1
        # self.max_cnt.append(self.dev_cnt)
        return self.vertices, self.num_vertices

    def add_edges(self, tokens):
        st = tokens[0].lower()
        if st.startswith('r') or st.startswith('c') or st.startswith('l'):
            e1 = tokens[1].lower()
            e2 = tokens[2].lower()
            # if e2 not in self.edges:
            self.edges.add((e1, e2))
            self.wires[st] = [e1, e2]
            if st.startswith('r'):
                rval = tokens[3]
                self.features[st] = [rval]
            elif st.startswith('c'):
                cval = tokens[3]
                self.features[st] = [cval]
            elif st.startswith('l'):
                lval = tokens[3]
                self.features[st] = [lval]

        elif st.startswith('m'):
            tp = tokens[5].lower()
            if tp.startswith('p'):
                tok = "p" + st
            else:
                tok = "n" + st
            e_d = tokens[1].lower()
            e_g = tokens[2].lower()
            e_src = tokens[3].lower()
            e_sub = tokens[4].lower()
            mos_tp = tokens[5].lower()
            self.edges.add((e_sub, e_g))
            self.edges.add((e_d, e_sub))
            self.edges.add((e_src, e_sub))
            self.wires[tok] = [e_d, e_g, e_src, e_sub]
            self.features[tok] = [tokens[6], tokens[7]]
        # elif st.startswith('vin') or st.startswith('i') or st.startswith('vout') or st.startswith('vinput') or st.startswith('vdd') or st.startswith('vss'):
        elif st.startswith('v') or st.startswith('i'):
            if st != 'vout':
                e1 = tokens[1].lower()
                e2 = tokens[2].lower()
                # if e2 not in self.edges:
                self.edges.add((e1, e2))
                self.wires[st] = [e1, e2]
                self.features[st] = [tokens[3]]
            elif st == 'vout':
                pass

        return self.edges, self.wires, self.features

    def is_source(self, xid):
        return xid in self.devices and self.devices[xid][cattr.DEVICE].is_source()

    def is_device(self, xid):
        return not self.is_source(xid)

    def gen_features(self, tokens):
        val = 0
        devtp = None
        cnt = 0
        is_unk = 0

        for key in self.nodenm2id:
            # print('key node name  : ', key)
            if key.startswith("x") and len(key)>2:
                devtp = devtype.INPUT
                cnt = self.comp2index(key)
                val = self.vin_val

            elif key.startswith("y"):
                devtp = devtype.OUTPUT
                cnt = self.comp2index(key)
                val = 0

            elif key == "vdd":
                continue

            elif key == "0":
                devtp = devtype.GND
                cnt = self.comp2index(key)
                val = 0

            elif key != "0" and key.isnumeric():
                devtp = devtype.INTER
                cnt = self.comp2index(key)
                val = 0

            self.feat_vect.setdefault('Component ID', []).append(cnt)
            self.feat_vect.setdefault('Component Type', []).append(devtp)
            self.feat_vect.setdefault('Features').append([0, 0, 0, 0, 0, is_unk, self.vin_val])
            # self.feat_vect['Features'] += [self.vout_val]
            self.feat_vect.setdefault('Vout', []).append(self.vout_val)

        return


def main(b_file: str):
    print("Parsing the file: ", b_file)
    acir = AnaCircuit(b_file)
    acir.read_bench(b_file)
    return


if __name__ == "__main__":
    if len(sys.argv) == 2:
        ben_file = sys.argv[1]
        main(ben_file)
    print("No arguments passed: pass the bench file ")

