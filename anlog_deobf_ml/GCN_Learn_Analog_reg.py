# __________________________
## import basic packages
# __________________________
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.sparse as sp
from scipy.stats import gmean
import time, datetime, resource, argparse
import warnings

warnings.filterwarnings('ignore')
import os, sys
import pickle, re, copy

# __________________________
## import internal py files
# __________________________
import to_graph_nx_rev
import graph_anadec

# __________________________
## import sklearn packages
# __________________________
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import SGDRegressor, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

# __________________________
# import tensorflow and keras packages
# __________________________
import tensorflow as tf
from tensorflow.keras import layers, Model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(gpus[0], [
            tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5000)])
    except RuntimeError as e:
        print(e)
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# __________________________
# import Spekctral package
# __________________________
import spektral
from spektral.data import Dataset, Graph, Loader, utils
from spektral.data.loaders import DisjointLoader
from spektral.layers import GlobalSumPool, GlobalAvgPool
from spektral.layers import GCSConv, GCNConv


def get_memory():
    with open('/proc/meminfo', 'r') as mem:
        free_memory = 0
        for i in mem:
            sline = i.split()
            if str(sline[0]) in ('MemFree:', 'Buffers:', 'Cached:'):
                free_memory += int(sline[1])
    return free_memory


def limit_memory(maxsize):
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    print(soft, hard)
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))


class AnaGraphDataset(Dataset):
    """
    A dataset for circuits in bench format
    """

    def __init__(self, bench_dir, **kwargs):

        # Create graphs from netlist extraction and assign feature vector to each node after normalizing the features

        self.colors = {'RES': 'cyan', 'CAP': 'green', 'IND': 'blue', 'PMOS': 'yellow', 'NMOS': 'gray'}
        self.indices = {'URES1': 0, 'UCAP1': 1, 'UIND1': 2, 'UPMOS1': 3, 'UNMOS1': 4,
                        'URES2': 5, 'UCAP2': 6, 'UIND2': 7, 'UPMOS2': 8, 'UNMOS2': 9,
                        'URES3': 10, 'UCAP3': 11, 'UIND3': 12, 'UPMOS3': 13, 'UNMOS3': 14,
                        'URES4': 15, 'UCAP4': 16, 'UIND4': 17, 'UPMOS4': 18, 'UNMOS4': 19,
                        'UPMOS5': 20, 'UNMOS5': 21,
                        'UPMOS6': 22, 'UNMOS6': 23}
        self.label_names = ['URES1', 'UCAP1', 'UIND1', 'UPMOS1', 'UNMOS1',
                            'URES2', 'UCAP2', 'UIND2', 'UPMOS2',  'UNMOS2',
                            'URES3', 'UCAP3', 'UIND3', 'UPMOS3', 'UNMOS3',
                            'URES4', 'UCAP4', 'UIND4', 'UPMOS4', 'UNMOS4',
                            'UPMOS5', 'UNMOS5'
                            'UPMOS6', 'UNMOS6']
        self.num_classes = len(self.indices)
        self.num_nodes = None
        self.num_epochs = 150
        self.batch_size = 50
        self.patience = 0
        self.learning_rate = 0.001
        self.df_features = pd.DataFrame()
        self.df_target = pd.DataFrame()
        self.X = []
        self.Y = []
        self.max_cnt = []
        self.maxcount = 0
        self.max_m_count = 0
        self.max_mos_cnt = []

        # load or generate the data
        operation = args.operation
        self.no_of_unkn = args.no_of_unkn
        if operation == 'train':
            self.dill_path = './data/spektral/train/ml_{}_{}_{}/'.format(args.network_type, args.name_of_structure,
                                                               args.no_of_unkn)
        elif operation == 'test':
            self.dill_path = './data/spektral/test/ml_{}_{}_{}/'.format(args.network_type, args.name_of_structure,
                                                               args.no_of_unkn)
        else:
            raise Exception('incorrect operation')

        self.dill_filename = 'data.dill'

        try:
            os.makedirs(self.dill_path)
        except:
            pass

        self.dill_file = self.dill_path + self.dill_filename

        if os.path.exists(self.dill_file):
            # load from dill file
            with open(self.dill_file, 'rb') as dill_fn:
                self.graphs = pickle.load(dill_fn)
        else:
            self.bench_dir = bench_dir
            self.graphs = []

            i = 1
            num_cirs = len(os.listdir(self.bench_dir))
            for bench_name in os.listdir(self.bench_dir):
                print('processing {0} {1}/{2}'.format(bench_name, i, num_cirs))
                bench_file = bench_dir + bench_name
                G = self.file2graph(bench_file)
                self.graphs.append(G)
                i += 1

            # save model to dill file
            with open(self.dill_file, 'wb') as dill_fn:
                pickle.dump(self.graphs, dill_fn)
            print('done writing to dill')

        super().__init__(**kwargs)

    def read(self):
        return self.graphs

    ## Create Graphs (Adjacency matrix, node features, value of unknown parameter) using Spektral
    ## from circuit features for training GNN model (RF, DT, XGB, NN)
    def file2graph(self, bench_file):
        cir = graph_anadec.AnaCircuit(bench_file)
        bench_name = os.path.basename(bench_file)
        bench_dill = os.path.join(self.dill_path, bench_name.replace('.bench', '.dill'))
        print('bench dill ', bench_dill)
        if os.path.exists(bench_dill):
            print('loading graph from ', bench_dill)
            with open(bench_dill, 'rb') as dill_fn:
                G = pickle.load(dill_fn)
            return G

        # Generate Graph from netlist
        vert, wires, feat, ckt_name, x, y = cir.read_bench(bench_file)
        Gh = to_graph_nx_rev.CreateGraph(vert, wires, feat)

        # Create Node Features Vector (X)
        feature_vect = cir.feat_vect
        # print("Feature vector: ", feature_vect)
        network_type = args.network_type

        Comp_id = []
        Comp_type = []
        Comp_Features = []
        Comp_vout = []

        for key, value in feature_vect.items():
            if key == "Component ID":
                Comp_id = value
            elif key == "Component Type":
                Comp_type = value
            elif key == "Vout":
                Comp_vout = value
            else:
                Comp_Features = value

        self.num_nodes = len(Comp_id)
        le = preprocessing.LabelEncoder()
        Comp_id = (le.fit_transform(Comp_id))
        Comp_type = (le.fit_transform(Comp_type))

        print("num_nodes: ", self.num_nodes)
        X = np.zeros((self.num_nodes, 14))

        feature_values = []
        for i in range(len(Comp_Features)):
            feature_values.append(Comp_type[i])
            feature_values.append(Comp_id[i])
            for v in range(len(Comp_Features[i])):
                feature_values.append(Comp_Features[i][v])
            for x in range(len(Comp_vout[i])):
                feature_values.append(Comp_vout[i][x])

        x = feature_values = np.array(feature_values).reshape(self.num_nodes, 14)

        if network_type == 'Resistor' or network_type == 'RC' or network_type=='ALPF_1ST' \
                or network_type=='ALPF_2ND' or network_type=='ABPF_4TH' or network_type=='ota_bpf':
            if self.no_of_unkn == 1:
                x = feature_values = np.array(feature_values).reshape(self.num_nodes, 14)
                D1 = list(range(9, 14))
                x = x[:, D1]
                x = x.astype(np.float32)
                x = x * 1e4

            elif network_type == 'RC' and self.no_of_unkn == 2:
                x = np.array(feature_values).reshape(self.num_nodes, 14)
                D1 = list(range(0, 1)) + list(range(5, 14))
                x = x[:, D1]

            elif network_type == 'ALPF_1ST' or network_type == 'ALPF_2ND' and self.no_of_unkn == 2:
                x = feature_values = np.array(feature_values).reshape(self.num_nodes, 14)

                D1 = list(range(0, 1)) + list(range(2, 6)) + list(range(7, 14))
                print('D1:', D1)
                x = x[:, D1]
                x = x.astype(np.float32)
                print('pre_sc:', x)

            elif network_type == 'ABPF_4TH' and self.no_of_unkn == 3 or self.no_of_unkn == 4:
                D1 = list(range(0, 1)) + list(range(3, 4)) + list(range(7, 8)) + list(range(9, 14))
                print('D1:', D1)
                x = x[:, D1]
                x = x.astype(np.float32)
                x[:, 1:2] = x[:, 1:2] * 1e8
                x[:, 3:7] = x[:, 3:7] * 1e4
                x[:, 7:8] = x[:, 7:8] * 1e-2
            elif network_type == 'ota_bpf' and self.no_of_unkn == 6:
                D1 = list(range(9, 14))
                x = x[:, D1]
                x = x.astype(np.float32)
                x = x * 1e4
        elif network_type == 'CCM' or network_type == 'LC_Osc' or network_type == 'TWG':
        # #     x = feature_values = np.array(feature_values).reshape(self.num_nodes, 8)

        # # LC_OSC Scaling
            if network_type == 'LC_Osc':
                if self.no_of_unkn == 1:
                    x[:, 1:5] = x[:, 1:5] * 1e3
                    x[:, 5:6] = x[:, 5:6] * 1e-5
                if self.no_of_unkn == 2:
                    x[:, 1:5] = x[:, 1:5] * 1e3
                    x[:, 5:6] = x[:, 5:6] * 1e-4
                if self.no_of_unkn == 3:
                    D1 = list(range(0, 1)) + list(range(3, 4)) + list(range(7, 8)) + list(range(9, 10)) + list(range(13,14))
                    print('D1:', D1)
                    x = x[:, D1]
                    x = x.astype(np.float32)
                    x[:, 1:2] = x[:, 1:2] * 1e12
                    x[:, 3:4] = x[:, 3:4] * 1e3
                    x[:, 4:5] = x[:, 4:5] * 1e-5


        # # CCM Scaling
            elif network_type == 'CCM':
                x[:, 1:2] = x[:, 1:2]*1e5
                x[:, 2:3] = x[:, 2:3]*1e4
                x[:, 4:6] = x[:, 4:6]*1e4

        # TWG Scaling
            elif network_type == 'TWG':
                if self.no_of_unkn == 1:
                    x[:, 4:5] = x[:, 4:5] * 1e1
                    x[:, 5:6] = x[:, 5:6]*1e-4
                elif self.no_of_unkn == 2:
                    x[:, 5:6] = x[:, 5:6] * 1e-4

        X = x.astype(np.float32)
        # print('post scale', X)
        # print('x shape :', X.shape)
        assert not np.any(np.isnan(X))

        # Create Adjacency matrix (A)
        A = sp.lil_matrix((self.num_nodes, self.num_nodes))

        pairs = [(key, value)
                 for key, values in wires.items()
                 for value in values]

        for pair in pairs:
            index1 = Gh.nm2index(pair[0])
            index2 = Gh.nm2index(pair[1])
            A[index1, index2] = 1

        # Create Labels (Y)
        unk_count = cir.unknown_count
        unk_comp = cir.unk_comp
        assert unk_count == len(unk_comp), f"components not equal to unknown count, got: {len(unk_comp)}"

        out_vect = cir.queries
        num_queries = unk_count
        if num_queries != 0:
            tot_queries = int(num_queries * 2)

        Y_val = np.empty((0, 1), float)
        Y = np.zeros(self.num_classes)

        com_id = out_vect.get("Component ID")
        query = out_vect.get("q_dev")
        y = query

        dtp = graph_anadec.devtype

        for i in range(1, num_queries + 1):
            if y[i - 1][0] == dtp.RES:
                label_names = 'URES' + str(i)
                Y[self.indices[label_names]] = 1
                val = y[i - 1][1]
                Y_val = np.append(Y_val, val)
            elif y[i - 1][0] == dtp.CAP:
                label_names = 'UCAP' + str(i)
                Y[self.indices[label_names]] = 1
                val = y[i - 1][1]
                Y_val = np.append(Y_val, val)
            elif y[i - 1][0] == dtp.IND:
                label_names = 'UIND' + str(i)
                Y[self.indices[label_names]] = 1
                val = y[i - 1][1]
                Y_val = np.append(Y_val, val)
            elif y[i - 1][0] == dtp.PMOS:
                label_names = 'UPMOS' + str(i)
                Y[self.indices[label_names]] = 1
                val = y[i - 1][1]
                Y_val = np.append(Y_val, val)
            elif y[i - 1][0] == dtp.NMOS:
                label_names = 'UNMOS' + str(i)
                Y[self.indices[label_names]] = 1
                val = y[i - 1][1]
                Y_val = np.append(Y_val, val)

        # print(Y_val.shape)
        if args.no_of_unkn == 2:
            if Y_val.shape == (1,):
                val = 0
                Y_val = np.append(Y_val, val)

        elif args.no_of_unkn == 3:
            if Y_val.shape == (1,):
                val = 0
                Y_val = np.append(Y_val, val)
            elif Y_val.shape == (2,):
                val = 0
                Y_val = np.append(Y_val, val, val)

        elif args.no_of_unkn == 4:
            if Y_val.shape == (1,):
                val = 0
                Y_val = np.append(Y_val, val)
            elif Y_val.shape == (2,):
                val = 0
                Y_val = np.append(Y_val, val, val)
            elif Y_val.shape == (3,):
                val = 0
                Y_val = np.append(Y_val, val, val, val)

        elif args.no_of_unkn == 6:
            if Y_val.shape == (1,):
                val = 0
                Y_val = np.append(Y_val, val)
            elif Y_val.shape == (2,):
                val = 0
                Y_val = np.append(Y_val, val, val)
            elif Y_val.shape == (3,):
                val = 0
                Y_val = np.append(Y_val, val, val, val)
            elif Y_val.shape == (4,):
                val = 0
                Y_val = np.append(Y_val, val, val, val, val)
            elif Y_val.shape == (5,):
                val = 0
                Y_val = np.append(Y_val, val, val, val, val)


        Y_val = Y_val.astype(np.float32)
        Y_val = np.log10(Y_val)
        # print('Y_val: ', Y_val)
        # print('y shape :', Y_val.shape)
        # print('X: ', X)

        G = Graph(a=A, x=X, y=Y_val)
        print(G)

        # exit(1)
        with open(bench_dill, 'wb') as dill_fn:
            pickle.dump(G, dill_fn)

        self.max_cnt.append(cir.dev_cnt)
        self.maxcount = max(self.max_cnt)
        print("max count :", self.maxcount)

        return G


### Create Dataframe from circuit features for training Vector machine learning models (RF, DT, XGB, NN)
    def build_dataframe(self, bench_dir):
        self.bench_dir = bench_dir

        i = 1
        num_cirs = len(os.listdir(self.bench_dir))
        for bench_name in os.listdir(self.bench_dir):
            # print('processing {0} {1}/{2}'.format(bench_name, i, num_cirs))
            bench_file = bench_dir + bench_name
            ga = graph_anadec.AnaCircuit(bench_file)
            dtp = graph_anadec.devtype
            vert, wires, feat, ckt_name, x_features, y_target = ga.read_bench(bench_file)

            # find network with maximum device count for the given data

            print("max count ::", self.maxcount)

            # Find number of known devices by subtracting unknown count
            kn_cnt = ga.dev_cnt - ga.unknown_count
            print("Known count :", kn_cnt)
            print("Device count :", ga.dev_cnt)
            print("unknown count :", ga.unknown_count)

            feature_vect = ga.feat_vect

            Comp_type = []

            for key, value in feature_vect.items():
                if key == "Component Type":
                    Comp_type = value

            mos_cnt = 0
            for comp in Comp_type:
                if comp == dtp.PMOS or comp == dtp.NMOS:
                    if comp not in ga.unk_comp:
                        mos_cnt += 1
                    else:
                        pass

            X_features = x_features
            Y_target = y_target

            self.X.append(X_features)
            self.Y.append(Y_target)

        max_lengths_f = {key: max(len(d[key]) for d in self.X) for key in self.X[0].keys()}

        columns = [f"{key}_{i + 1}" for key, max_length in max_lengths_f.items() for i in range(max_length)]
        self.df_features = pd.DataFrame(columns=columns).fillna(0)

        for idx, d in enumerate(self.X):
            for key, values in d.items():
                for i, value in enumerate(values):
                    column_name = f"{key}_{i + 1}"
                    self.df_features.at[idx, column_name] = value

        self.df_features.rename(columns={'Features_1': 'dev_cnt', 'Features_2': 'num_IO',
                                         'Features_3': 'num_res', 'Features_4': 'num_cap', 'Features_5': 'num_ind',
                                         'Features_6': 'num_pmos', 'Features_7': 'num_nmos',
                                         'Features_8': 'unknown_count',
                                         'Features_9': 'vin_volt_val', 'Features_10': 'vin_frq_val',
                                         'Features_11': 'vout1',
                                         'Features_12': 'vout2', 'Features_13': 'vout3', 'Features_14': 'vout4',
                                         'Features_15': 'vout5'}, inplace=True)

        self.df_features.replace(np.nan, 0, inplace=True)

        max_lengths = {key: max(len(d[key]) for d in self.Y) for key in self.Y[0].keys()}

        columns = [f"{key}_{i + 1}" for key, max_length in max_lengths.items() for i in range(max_length)]
        self.df_target = pd.DataFrame(columns=columns).fillna(0)

        for idx, d in enumerate(self.Y):
            for key, values in d.items():
                for i, value in enumerate(values):
                    column_name = f"{key}_{i + 1}"
                    self.df_target.at[idx, column_name] = value

        self.df_target.replace(np.nan, 0, inplace=True)

        return self.df_features, self.df_target


# _____________________________________________
# Build Machine Learning Class
# _____________________________________________

class Mlearning():
    def __init__(self, input_features, target, data, **kwargs):
        self.features = input_features
        self.target = target
        self.num_epochs = 150
        self.batch_size = 50
        self.patience = 30
        self.learning_rate = 0.001
        self.ml_method = 'RF'
        self.input_sample_size = 1.0
        self.std_scaler = StandardScaler()
        # self.std_scaler = MinMaxScaler()

        no_of_unkn = args.no_of_unkn
        name_of_struct = args.name_of_structure
        network_type = args.network_type
        df_name = args.df_name
        no_of_samples = args.no_of_samples
        ml_method = args.ml_method
        operation = args.operation
        sample_size = args.sample_size

        columns = ['Type of Dataset', 'Number of samples', 'Mean Percent Error']
        self.model_sens = pd.DataFrame(columns=columns)

        # if ml_method != 'GCN':
        circuit_df = self.process_data(input_features, target)
        print(circuit_df.head(5))
        print(f" Shape of the dataframe = {circuit_df.shape}")

        zero_cols = [col for col, is_zero in ((circuit_df == 0).sum() == circuit_df.shape[0]).items() if is_zero]
        # print(zero_cols)
        # exit(1)
        circuit_df.drop(zero_cols, axis=1, inplace=True)
        print(f" Shape of the dataframe AFTER DROPPING ZERO COLUMNS = {circuit_df.shape}")
        circuit_df.to_csv('~/PycharmProjects/pythonProject1/gcn_ana/circuit_df.csv')
        # exit(1)

        X, y = self.feature_struct(circuit_df, name_of_struct, network_type, no_of_unkn=no_of_unkn)
        # # summarize dataset
        print(X.shape, y.shape)

        X_copy = X.copy(deep=True)
        # impute zero values with the mean value of the column
        X_copy.replace(0, X_copy.mean(axis=0), inplace=True)
        X_copy.to_csv('~/PycharmProjects/pythonProject1/gcn_ana/X_features.csv')
        # print(X_copy.sample(5))

        if operation == 'train':
            X_train_std, X_test_std, y_train_std, y_test_std = self.split_scale_data(X_copy, y, test_size=0.2)
            # print(X_train_std)

        elif operation == 'test' and ml_method != 'GCN':
            X_test = self.std_scaler.fit_transform(X_copy)
            y_test = self.std_scaler.fit_transform(y)

        elif operation == 'test' and ml_method == 'GCN':
            loader_te = DisjointLoader(data)

        else:
            raise Exception('Invalid Operation')

        # exit(1)

        if operation == 'train':
            df1 = self.call_model_train(ml_method, data, X_train_std,
                                        X_test_std, y_train_std, y_test_std,
                                        no_of_unkn=no_of_unkn)
            print(df1)

        elif operation == 'test' and ml_method != 'PLOT' and ml_method != 'GCN':
            df1 = self.test_model(ml_method, X_test, y_test)
            print(df1)

        elif operation == 'test' and ml_method == 'GCN':
            file_path = './data/model/'
            if not os.path.isdir(file_path):
                os.makedirs(file_path)
            filename = os.path.join(file_path, 'ml_{}_{}_{}_{}'.format('GCN', args.network_type,
                                                                       args.name_of_structure,
                                                                       args.no_of_unkn))
            loader_te = DisjointLoader(data)
            df1 = self.gcn_test(loader_te, filename)
            df_name = self.clean_dataframe(df1)
            self.save_csv_file(df_name, 'GCN')
            print(df_name)

        elif operation == 'test' and ml_method == 'PLOT':
            self.call_plotting_data()

        print('success')

    def call_model_train(self, ml_method, data, X_train_std, X_test_std, y_train_std, y_test_std, no_of_unkn):
        no_of_samples = args.no_of_samples
        no_of_unkn = args.no_of_unkn
        df = pd.DataFrame()
        match ml_method:
            case 'LR':
                df = self.train_and_predict_ml_LR(X_train_std, X_test_std, y_train_std, y_test_std,
                                                  no_of_unkn=no_of_unkn, no_of_samples=no_of_samples)
            case 'SGD':
                df = self.train_and_predict_ml_SGD(X_train_std, X_test_std, y_train_std, y_test_std,
                                                   no_of_unkn=no_of_unkn, no_of_samples=no_of_samples)
            case 'DT':
                df = self.train_and_predict_ml_DT(X_train_std, X_test_std, y_train_std, y_test_std,
                                                  no_of_unkn=no_of_unkn, no_of_samples=no_of_samples)
            case 'RF':
                df = self.train_and_predict_ml_RF(X_train_std, X_test_std, y_train_std, y_test_std,
                                                  no_of_unkn=no_of_unkn, no_of_samples=no_of_samples)
            case 'XGB':
                df = self.train_and_predict_ml_XGB(X_train_std, X_test_std, y_train_std, y_test_std,
                                                   no_of_unkn=no_of_unkn, no_of_samples=no_of_samples)
            case 'NN':
                df = self.train_and_predict_ml_NN(X_train_std, X_test_std, y_train_std, y_test_std,
                                                  no_of_unkn=no_of_unkn, no_of_samples=no_of_samples)
            case 'GCN':
                df = self.train_and_predict_ml_GCN(data, no_of_unkn=no_of_unkn, no_of_samples=no_of_samples)
        return df

    def df_concat_for_plots(self, *kwargs):
        operation = args.operation

        # df1 = self.load_csv_file('LR')
        df2 = self.load_csv_file('SGD')
        df3 = self.load_csv_file('DT')
        df4 = self.load_csv_file('RF')
        df5 = self.load_csv_file('XGB')
        df6 = self.load_csv_file('NN')
        df7 = self.load_csv_file('GCN')

        result_df = pd.concat([df2, df3, df4, df5, df6, df7], axis=0, ignore_index=True)
        # result_df = pd.concat([df1, df3, df4, df5, df6], axis=0, ignore_index=True)

        if operation == 'train':

            file_path = './data/Results/train/ml_{}_{}_{}/'.format(args.network_type,
                                                         args.name_of_structure,
                                                         args.no_of_unkn)
        elif operation == 'test':

            file_path = './data/Results/test/ml_{}_{}_{}/'.format(args.network_type,
                                                         args.name_of_structure,
                                                         args.no_of_unkn)
        else:
            raise Exception('Invalid Operation')

        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        to_file = os.path.join(file_path, 'result_df.csv')
        result_df.to_csv(to_file)

        return result_df
        # return df2, df3, df4, df5, df6, df7

    def process_data(self, input_features, target):
        circuit_df = pd.concat([input_features, target], axis=1)
        return circuit_df

    def feature_struct(self, circuit_df, name_of_struct, network_type, no_of_unkn):
        circuit_X, circuit_Y = pd.DataFrame(), pd.DataFrame()

        # ================Generate FEATURES: INPUT  - OUTPUT  ====================
        # ------------- for RESISTOR NETWORK -----------
        if name_of_struct == "i_o" and network_type == 'Resistor':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1']]
                circuit_Y = circuit_df[['UN_R_Val_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1']]
                circuit_Y = circuit_df[['UN_R_Val_1', 'UN_R_Val_2']]

        # ---------------- for RC NETWORK ------------------
        elif name_of_struct == "i_o" and network_type == 'RC':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_C_Val_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_R_Val_1', 'UN_C_Val_2']]

        # -------------- for Mixed Analog NETWORK ----------------
        elif name_of_struct == "i_o" and network_type == 'Analog':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_MOS_LN_1', 'UN_MOS_WD_1' 'UN_R_Val_1',
                                        'UN_C_Val_1', 'UN_L_Val_1']]

            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_MOS_LN_1', 'UN_MOS_LN_2', 'UN_MOS_WD_1', 'UN_MOS_WD_2',
                                        'UN_R_Val_1', 'UN_R_Val_2', 'UN_C_Val_1', 'UN_C_Val_2',
                                        'UN_L_Val_1', 'UN_L_Val_2']]

        # ---------------- LC Oscillator ------------------
        elif name_of_struct == "i_o" and network_type == 'LC_Osc':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_L_Val_2']]

        # ---------------- Constant Current Mirror ------------------
        elif name_of_struct == "i_o" and network_type == 'CCM':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1', 'vout2']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1', 'vout2']]
                circuit_Y = circuit_df[['UN_MOS_LN_1', 'UN_MOS_WD_1']]

        # ---------------- Triangular Wave Generator ------------------
        elif name_of_struct == "i_o" and network_type == 'TWG':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1']]
                circuit_Y = circuit_df[['UN_MOS_LN_1', 'UN_MOS_WD_1']]

        # =============== for FEATURES: INPUT  - OUTPUT - KNOWN VALUES =================#
        # ------------- for RESISTOR NETWORK -----------
        elif name_of_struct == "i_o_val" and network_type == 'Resistor':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'R_Val_1', 'R_Val_2', 'R_Val_3']]
                circuit_Y = circuit_df[['UN_R_Val_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'R_Val_1', 'R_Val_2', 'R_Val_3']]
                circuit_Y = circuit_df[['UN_R_Val_1', 'UN_R_Val_2']]

        # ---------------- for RC NETWORK ------------------
        elif name_of_struct == "i_o_val" and network_type == 'RC':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'R_Val_1']]
                circuit_Y = circuit_df[['UN_C_Val_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'R_Val_1']]
                circuit_Y = circuit_df[['UN_C_Val_1', 'UN_C_Val_2']]

        # -------------- for Mixed Analog NETWORK ----------------
        elif name_of_struct == "i_o_val" and network_type == 'Analog':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'MOS_LN_1', 'MOS_WD_1', 'R_Val_1', 'C_Val_1',
                                        'L_Val_1']]
                circuit_Y = circuit_df[['UN_MOS_LN_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'MOS_LN_1', 'MOS_WD_1', 'R_Val_1', 'C_Val_1',
                                        'L_Val_1']]
                circuit_Y = circuit_df[['UN_MOS_LN_1', 'UN_MOS_LN_2', 'UN_MOS_WD_1', 'UN_MOS_WD_2',
                                        'UN_R_Val_1', 'UN_R_Val_2', 'UN_C_Val_1', 'UN_C_Val_2',
                                        'UN_L_Val_1', 'UN_L_Val_2']]

        # ---------------- LC Oscillator ------------------
        elif name_of_struct == "i_o_val" and network_type == 'LC_Osc':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1', 'MOS_LN_4', 'MOS_WD_4',
                                        'MOS_LN_7', 'MOS_WD_7',
                                        'MOS_LN_10', 'MOS_WD_10',
                                        'MOS_LN_20', 'MOS_WD_20']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1', 'MOS_LN_4', 'MOS_WD_4',
                                        'MOS_LN_7', 'MOS_WD_7',
                                        'MOS_LN_10', 'MOS_WD_10',
                                        'MOS_LN_20', 'MOS_WD_20']]
                circuit_Y = circuit_df[['UN_MOS_LN_1', 'UN_MOS_WD_1']]

        # ---------------- Constant Current Mirror ------------------
        elif name_of_struct == "i_o_val" and network_type == 'CCM':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'MOS_LN_2', 'MOS_WD_2', 'MOS_LN_3', 'MOS_WD_3',
                                        'MOS_LN_4', 'MOS_WD_4',
                                        'R_Val_3', 'R_Val_5']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'MOS_LN_2', 'MOS_WD_2', 'MOS_LN_3', 'MOS_WD_3',
                                        'MOS_LN_4', 'MOS_WD_4',
                                        'R_Val_3', 'R_Val_5']]
                circuit_Y = circuit_df[['UN_MOS_LN_1', 'UN_MOS_WD_1']]

        # ---------------- Triangular Wave Generator ------------------
        elif name_of_struct == "i_o_val" and network_type == 'TWG':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1', 'vout5', 'MOS_LN_4', 'MOS_WD_4',
                                        'MOS_LN_7', 'MOS_WD_7']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1', 'vout5', 'MOS_LN_4', 'MOS_WD_4',
                                        'MOS_LN_7', 'MOS_WD_7']]
                circuit_Y = circuit_df[['UN_MOS_LN_1', 'UN_MOS_WD_1']]

        # =================== for FEATURES: INPUT  - OUTPUT - KNOWN VALUES - TOPOLOGY =================
        # ------------- for RESISTOR NETWORK -----------
        elif name_of_struct == "i_o_val_topo" and network_type == 'Resistor':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1',
                                        'R_Val_1', 'R_Val_2', 'R_Val_3',
                                        'dev_cnt', 'num_IO', 'num_res']]
                circuit_Y = circuit_df[['UN_R_Val_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1',
                                        'R_Val_1', 'R_Val_2',
                                        'dev_cnt', 'num_IO', 'num_res']]
                circuit_Y = circuit_df[['UN_R_Val_1', 'UN_R_Val_2']]

            # ---------------- for RC NETWORK ------------------
        elif name_of_struct == "i_o_val_topo" and network_type == 'RC':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'R_Val_1',
                                        'dev_cnt', 'num_IO', 'num_res', 'num_cap']]
                circuit_Y = circuit_df[['UN_C_Val_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_R_Val_1', 'UN_C_Val_2']]

        # -------------- for Active LPF 1ST ORDER ----------------
        elif name_of_struct == "i_o_val_topo" and network_type == 'ALPF_1ST':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_R_Val_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_R_Val_1', 'UN_C_Val_2']]
            elif no_of_unkn == 3:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_R_Val_1', 'UN_C_Val_2']]
            elif no_of_unkn == 4:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_R_Val_1', 'UN_C_Val_2']]

        # -------------- for Active LPF 2ND ORDER ----------------
        elif name_of_struct == "i_o_val_topo" and network_type == 'ALPF_2ND':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                    'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_R_Val_1']]
            elif no_of_unkn == 2:
                # circuit_X = circuit_df[[ 'vout1', 'vout2', 'vout3',
                #                         'vout4', 'vout5','R_Val_30','C_Val_31']]
                # circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                #                         'vout4', 'vout5', 'R_Val_29', 'C_Val_31']]
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'C_Val_31']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_R_Val_2']]
                # circuit_Y = circuit_df[['UN_R_Val_1',
                #                         'UN_C_Val_2']]
            elif no_of_unkn == 3:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5','R_Val_30','C_Val_31']]
                circuit_Y = circuit_df[['UN_R_Val_1',
                                        'UN_C_Val_2']]
            elif no_of_unkn == 4:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5','R_Val_30','C_Val_31']]
                circuit_Y = circuit_df[['UN_R_Val_1',
                                        'UN_C_Val_2']]

                # -------------- for Active BPF 4TH ORDER ----------------
        elif name_of_struct == "i_o_val_topo" and network_type == 'ABPF_4TH':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vin_volt_val', 'vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_R_Val_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'R_Val_30', 'C_Val_31']]
                circuit_Y = circuit_df[['UN_R_Val_1',
                                        'UN_C_Val_2']]
            elif no_of_unkn == 3:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'C_Val_64', 'C_Val_67']]
                circuit_Y = circuit_df[['UN_R_Val_1','UN_R_Val_2',
                                        'UN_R_Val_3']]
            elif no_of_unkn == 4:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'R_Val_60', 'C_Val_66']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_R_Val_2','UN_R_Val_3',
                                        'UN_R_Val_4']]

            # -------------- for OTA BPF 4TH ORDER ----------------
        elif name_of_struct == "i_o_val_topo" and network_type == 'ota_bpf':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'UN_MOS_WD_2',
                                        'UN_MOS_WD_3',
                                        'UN_MOS_WD_4']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5', 'UN_MOS_WD_3',
                                        'UN_MOS_WD_4']]
                circuit_Y = circuit_df[['UN_MOS_WD_1',
                                        'UN_MOS_WD_2']]
            elif no_of_unkn == 3:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5',
                                        'UN_MOS_WD_4']]
                circuit_Y = circuit_df[['UN_MOS_WD_1','UN_MOS_WD_2',
                                        'UN_MOS_WD_3']]
            elif no_of_unkn == 4:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_MOS_WD_1','UN_MOS_WD_2',
                                        'UN_MOS_WD_3',
                                        'UN_MOS_WD_4']]
            elif no_of_unkn == 6:
                circuit_X = circuit_df[['MOS_WD_5', 'MOS_LN_5','MOS_WD_14','MOS_LN_14',
                                        'MOS_WD_23', 'MOS_LN_23', 'MOS_WD_42', 'MOS_LN_42',
                                        'MOS_WD_61', 'MOS_LN_61', 'MOS_WD_70', 'MOS_LN_70',
                                        'vout1', 'vout2', 'vout3', 'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_MOS_WD_1','UN_MOS_WD_2',
                                        'UN_MOS_WD_3','UN_MOS_WD_4',
                                        'UN_MOS_WD_5','UN_MOS_WD_6']]

        # ---------------- LC Oscillator ------------------
        elif name_of_struct == "i_o_val_topo" and network_type == 'LC_Osc':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1', 'vout5']]
                # circuit_X = circuit_df[['vout1', 'vout5', 'C_Val_35',
                #                         'L_Val_34']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1', 'vout5']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_L_Val_2']]
            elif no_of_unkn == 3:
                circuit_X = circuit_df[['C_Val_33', 'vout1', 'vout5']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_MOS_WD_2', 'UN_L_Val_3']]
            elif no_of_unkn == 4:
                circuit_X = circuit_df[['vout1', 'vout5']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_L_Val_2']]

        # ---------------- Constant Current Mirror ------------------
        elif name_of_struct == "i_o_val_topo" and network_type == 'CCM':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1', 'vout2', 'vout3',
                                        'MOS_LN_2', 'MOS_WD_2', 'MOS_LN_3', 'MOS_WD_3',
                                        'MOS_LN_4', 'MOS_WD_4',
                                        'R_Val_3', 'R_Val_5',
                                        'dev_cnt',
                                        'num_nmos']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout2', 'vout3' ]]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_MOS_WD_2']]
            elif no_of_unkn == 3:
                circuit_X = circuit_df[['vout2', 'vout3']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_MOS_WD_2']]
            elif no_of_unkn == 4:
                circuit_X = circuit_df[['vout2', 'vout3']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_MOS_WD_2']]

        # ---------------- Triangular Wave Generator ------------------
        elif name_of_struct == "i_o_val_topo" and network_type == 'TWG':
            if no_of_unkn == 1:
                circuit_X = circuit_df[['vout1', 'vout4', 'vout5']]
                circuit_Y = circuit_df[['UN_MOS_WD_1']]
            elif no_of_unkn == 2:
                circuit_X = circuit_df[['vout1', 'vout4', 'vout5', 'MOS_LN_4', 'MOS_WD_4',
                                        'MOS_LN_7', 'MOS_WD_7']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_MOS_WD_2']]
            elif no_of_unkn == 3:
                circuit_X = circuit_df[['vout1', 'vout4', 'vout5', 'MOS_LN_4', 'MOS_WD_4',
                                        'MOS_LN_7', 'MOS_WD_7']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_MOS_WD_2']]
            elif no_of_unkn == 4:
                circuit_X = circuit_df[['vout1', 'vout4', 'vout5', 'MOS_LN_4', 'MOS_WD_4',
                                        'MOS_LN_7', 'MOS_WD_7']]
                circuit_Y = circuit_df[['UN_MOS_WD_1', 'UN_MOS_WD_2']]

        return circuit_X, circuit_Y

    def split_scale_data(self, x, y, seed=7, test_size=0.3, sample_size=1.0):

        sample_size = args.sample_size

        print('preshape:', x.shape)
        xrows = int(x.shape[0] * sample_size)
        print('rows ', xrows)
        x = x.iloc[:xrows]
        y = y.iloc[:xrows]
        self.input_sample_size = xrows
        print('postshape:', x.shape)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=seed, shuffle=True)

        X_train_std = self.std_scaler.fit_transform(X_train)
        X_test_std = self.std_scaler.transform(X_test)
        y_train_std = self.std_scaler.fit_transform(y_train)
        y_test_std = self.std_scaler.transform(y_test)

        print(X_train_std.shape)
        print(X_test_std.shape)
        print(y_train_std.shape)
        print(y_test_std.shape)

        return X_train_std, X_test_std, y_train_std, y_test_std

    def print_save_result(self, model_name, no_of_samples, y_predict, y_test, Duration, no_of_unkn):
        df_name = self.create_df(no_of_unkn)
        # name_of_struct = args.name_of_struct
        pe_rf = []
        file_path = './data/Results/train/'
        to_file = os.path.join(file_path, f'model_sen_RF_{args.network_type}.csv')

        for i in range(no_of_samples):
        # for i in range(90):
            MSE = mean_squared_error(y_test[i], y_predict[i], squared=False)
            RMSE = np.sqrt(mean_absolute_error(y_test[i], y_predict[i]))
            if no_of_unkn == 1:
                percent_error = (((abs(y_test[i] - y_predict[i])) / y_test[i]) * 100)
                df_name.loc[len(df_name)] = model_name, y_test[i], \
                    y_predict[i], percent_error, MSE, RMSE, Duration
                pe_rf.append(percent_error)
            elif no_of_unkn == 2:
                percent_error_0 = (((abs(y_test[i][0] - y_predict[i][0])) / y_test[i][0]) * 100)
                percent_error_1 = (((abs(y_test[i][1] - y_predict[i][1])) / y_test[i][1]) * 100)
                # print("X = %s, Predicted =% s,percent_error = %s, MSE = %s, MAE = %s" %
                #       (y_test[i], y_predict[i], percent_error, MSE, MAE))
                df_name.loc[len(df_name)] = model_name, y_test[i][0], y_test[i][1], \
                    y_predict[i][0], y_predict[i][1], \
                    percent_error_0, percent_error_1, \
                    MSE, RMSE, Duration
            elif no_of_unkn == 3:
                percent_error_0 = (((abs(y_test[i][0] - y_predict[i][0])) / y_test[i][0]) * 100)
                percent_error_1 = (((abs(y_test[i][1] - y_predict[i][1])) / y_test[i][1]) * 100)
                percent_error_2 = (((abs(y_test[i][2] - y_predict[i][2])) / y_test[i][2]) * 100)
                # print("X = %s, Predicted =% s,percent_error = %s, MSE = %s, MAE = %s" %
                #       (y_test[i], y_predict[i], percent_error, MSE, MAE))
                df_name.loc[len(df_name)] = model_name, y_test[i][0], y_test[i][1], y_test[i][2], \
                    y_predict[i][0], y_predict[i][1], y_predict[i][2],\
                    percent_error_0, percent_error_1, percent_error_2,\
                    MSE, RMSE, Duration
            elif no_of_unkn == 4:
                percent_error_0 = (((abs(y_test[i][0] - y_predict[i][0])) / y_test[i][0]) * 100)
                percent_error_1 = (((abs(y_test[i][1] - y_predict[i][1])) / y_test[i][1]) * 100)
                percent_error_2 = (((abs(y_test[i][2] - y_predict[i][2])) / y_test[i][2]) * 100)
                percent_error_3 = (((abs(y_test[i][3] - y_predict[i][3])) / y_test[i][3]) * 100)
                # print("X = %s, Predicted =% s,percent_error = %s, MSE = %s, MAE = %s" %
                #       (y_test[i], y_predict[i], percent_error, MSE, MAE))
                df_name.loc[len(df_name)] = model_name, y_test[i][0], y_test[i][1],  y_test[i][2],  y_test[i][3],\
                    y_predict[i][0], y_predict[i][1], y_predict[i][2], y_predict[i][3],\
                    percent_error_0, percent_error_1, percent_error_2, percent_error_3, \
                    MSE, RMSE, Duration
            elif no_of_unkn == 6:
                percent_error_0 = (((abs(y_test[i][0] - y_predict[i][0])) / y_test[i][0]) * 100)
                percent_error_1 = (((abs(y_test[i][1] - y_predict[i][1])) / y_test[i][1]) * 100)
                percent_error_2 = (((abs(y_test[i][2] - y_predict[i][2])) / y_test[i][2]) * 100)
                percent_error_3 = (((abs(y_test[i][3] - y_predict[i][3])) / y_test[i][3]) * 100)
                percent_error_4 = (((abs(y_test[i][4] - y_predict[i][4])) / y_test[i][4]) * 100)
                percent_error_5 = (((abs(y_test[i][5] - y_predict[i][5])) / y_test[i][5]) * 100)
                # print("X = %s, Predicted =% s,percent_error = %s, MSE = %s, MAE = %s" %
                #       (y_test[i], y_predict[i], percent_error, MSE, MAE))
                df_name.loc[len(df_name)] = model_name, y_test[i][0], y_test[i][1],  y_test[i][2],  y_test[i][3], \
                    y_test[i][4], y_test[i][5], \
                    y_predict[i][0], y_predict[i][1], y_predict[i][2], y_predict[i][3], \
                    y_predict[i][4], y_predict[i][5], \
                    percent_error_0, percent_error_1, percent_error_2, percent_error_3, \
                    percent_error_4, percent_error_5, \
                    MSE, RMSE, Duration

        MPE = gmean(pe_rf)
        if model_name == 'RF':
            self.model_sens.loc[len(self.model_sens)] = args.network_type, self.input_sample_size, MPE
            print(self.model_sens)
            self.model_sens.to_csv(to_file)

        return df_name

    def save_csv_file(self, from_file, model_name):
        operation = args.operation
        if operation == 'train':
            file_path = './data/Results/train/ml_{}_{}_{}/'.format(args.network_type,
                                                                   args.name_of_structure,
                                                                   args.no_of_unkn)
        elif operation == 'test':
            file_path = './data/Results/test/ml_{}_{}_{}/'.format(args.network_type,
                                                                  args.name_of_structure,
                                                                  args.no_of_unkn)
        else:
            raise Exception('Invalid Operation')

        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        to_file = os.path.join(file_path, f'{model_name}.csv')

        from_file.to_csv(to_file)
        return

    def load_csv_file(self, model_name):
        operation = args.operation
        if operation == 'train':
            file_path = './data_f/Results/train/ml_{}_{}_{}/'.format(args.network_type,
                                                                   args.name_of_structure,
                                                                   args.no_of_unkn)
        elif operation == 'test':
            file_path = './data_f/Results/test/ml_{}_{}_{}/'.format(args.network_type,
                                                                  args.name_of_structure,
                                                                  args.no_of_unkn)
        else:
            raise Exception('Invalid Operation')

        if not os.path.isdir(file_path):
            os.makedirs(file_path)
        from_file = os.path.join(file_path, f'{model_name}.csv')
        to_file = pd.read_csv(from_file)
        return to_file

    def save_model(self, model_name, model, filename=None):

        final_path = './data/model/'
        if not os.path.isdir(final_path):
            os.makedirs(final_path)

        if filename is None:
            filename = './data/model/ml_{}_{}_{}_{}.txt'.format(model_name,
                                                                args.network_type,
                                                                args.name_of_structure,
                                                                args.no_of_unkn)

            pickle.dump(model, open(filename, 'wb'))
        print('\n---Model saved!---')

        return

    def load_saved_model(self, model_name, filename=None):
        final_path = './data/model/'
        if not os.path.isdir(final_path):
            os.makedirs(final_path)

        if filename is None:
            filename = './data/model/ml_{}_{}_{}_{}.txt'.format(model_name,
                                                                args.network_type,
                                                                args.name_of_structure,
                                                                args.no_of_unkn)

            loaded_model = pickle.load(open(filename, 'rb'))

        return loaded_model

## Create dataframe to save results

    def create_df(self, no_of_unkn):
        df_name = pd.DataFrame()
        if no_of_unkn == 1:
            df_name = pd.DataFrame(columns=["MODEL NAME", 'Actual value',
                                            'Predicted Value', 'percent_error',
                                            'MSE', 'RMSE', 'Duration'])
        elif no_of_unkn == 2:
            df_name = pd.DataFrame(
                columns=["MODEL NAME", 'Actual value0', 'Actual value1',
                         'Predicted Value0', 'Predicted Value1',
                         'percent_error_0', 'percent_error_1', 'MSE', 'RMSE', 'Duration'])
        elif no_of_unkn == 3:
            df_name = pd.DataFrame(
                columns=["MODEL NAME", 'Actual value0', 'Actual value1', 'Actual value2',
                         'Predicted Value0', 'Predicted Value1', 'Predicted Value2',
                         'percent_error_0', 'percent_error_1', 'percent_error_2', 'MSE', 'RMSE', 'Duration'])
        elif no_of_unkn == 4:
            df_name = pd.DataFrame(
                columns=["MODEL NAME", 'Actual value0', 'Actual value1', 'Actual value2', 'Actual value3',
                         'Predicted Value0', 'Predicted Value1', 'Predicted Value2', 'Predicted Value3',
                         'percent_error_0', 'percent_error_1', 'percent_error_2', 'percent_error_3', 'MSE', 'RMSE', 'Duration'])
        elif no_of_unkn == 6:
            df_name = pd.DataFrame(
                columns=["MODEL NAME", 'Actual value0', 'Actual value1', 'Actual value2', 'Actual value3',
                         'Actual value4', 'Actual value5',
                         'Predicted Value0', 'Predicted Value1', 'Predicted Value2', 'Predicted Value3',
                         'Predicted Value4', 'Predicted Value5',
                         'percent_error_0', 'percent_error_1', 'percent_error_2', 'percent_error_3',
                         'percent_error_4', 'percent_error_5',
                         'MSE', 'RMSE', 'Duration'])
        return df_name

    def test_model(self, model_name, X_test, Y_test):
        no_of_samples = args.no_of_samples
        no_of_unkn = args.no_of_unkn

        df_name = self.create_df(no_of_unkn)

        start = time.time()
        model = self.load_saved_model(model_name)
        y_predict = model.predict(X_test)

        if model_name == 'LR':
            y_predict = self.std_scaler.inverse_transform(y_predict)
            if no_of_unkn == 1:
                y_predict = y_predict.reshape(-1, 1)

            y_test = self.std_scaler.inverse_transform(Y_test)

        elif model_name == 'NN':
            y_predict = self.std_scaler.inverse_transform(y_predict)
            y_test = self.std_scaler.inverse_transform(Y_test)

        else:
            if no_of_unkn == 1:
                y_predict = y_predict.reshape(1, -1)

            y_predict = self.std_scaler.inverse_transform(y_predict)
            if no_of_unkn == 1:
                y_predict = y_predict.reshape(-1, 1)

            y_test = self.std_scaler.inverse_transform(Y_test)

        end = time.time()
        duration = end - start

        df_name = self.print_save_result(model_name, no_of_samples, y_predict, y_test, duration, no_of_unkn=no_of_unkn)
        df_name = self.clean_dataframe(df_name)

        self.save_csv_file(df_name, model_name)

        return df_name

    # +++++++++++++++++Linear Regressor++++++++++++++++++++++++++++
    def train_and_predict_ml_LR(self, X_train_std, X_test_std, y_train_std, y_test_std, no_of_unkn, no_of_samples=20):
        name = 'LR'
        # X_train_std, X_test_std, y_train_std, y_test_std = self.split_scale_data(X, y, seed=42, test_size=0.3)

        lnr = LinearRegression()

        lnr.fit(X_train_std, y_train_std)
        self.save_model(name, lnr)

        df_name = self.test_model(name, X_test_std, y_test_std)

        return df_name

        # +++++++++++++++++ SGD Regressor++++++++++++++++++++++++++++++

    def train_and_predict_ml_SGD(self, X_train_std, X_test_std, y_train_std, y_test_std, no_of_unkn, no_of_samples=20):
        name = 'SGD'
        from sklearn import linear_model
        # X_train_std, X_test_std, y_train_std, y_test_std = \
        #     self.split_scale_data(X, y, seed=52, test_size=0.3)
        if no_of_unkn == 1:
            sgd = SGDRegressor(alpha=0.0001, epsilon=0.01, eta0=0.1, penalty='elasticnet')

        elif no_of_unkn == 2 or no_of_unkn == 3 or no_of_unkn == 4 or no_of_unkn == 6:
            sgd = linear_model.LinearRegression()

        sgd.fit(X_train_std, y_train_std)
        self.save_model('SGD', sgd)

        df_name = self.test_model(name, X_test_std, y_test_std)

        return df_name

    # +++++++++++++++++Decision Treer++++++++++++++++++++++++++++
    def train_and_predict_ml_DT(self, X_train_std, X_test_std, y_train_std, y_test_std, no_of_unkn, no_of_samples=20):

        name = 'DT'

        # X_train_std, X_test_std, y_train_std, y_test_std = self.split_scale_data(X, y, seed=72, test_size=0.3)

        DT = DecisionTreeRegressor(ccp_alpha=0.0, criterion='squared_error', max_depth=None,
                                   max_features=None, max_leaf_nodes=None,
                                   min_impurity_decrease=0.0,
                                   min_samples_leaf=1, min_samples_split=2,
                                   min_weight_fraction_leaf=0.0, splitter='best', random_state=3)
        DT.fit(X_train_std, y_train_std)
        self.save_model(name, DT)

        df_name = self.test_model(name, X_test_std, y_test_std)

        return df_name

    # +++++++++++++++++Random Forest Regressor+++++++++++++++++++++++
    def train_and_predict_ml_RF(self, X_train_std, X_test_std, y_train_std, y_test_std, no_of_unkn, no_of_samples=20):

        name = 'RF'

        # X_train_std, X_test_std, y_train_std, y_test_std = self.split_scale_data(X, y, test_size=0.3)

        RF = RandomForestRegressor(n_estimators=1000, max_features='sqrt', max_depth=15, random_state=9)
        RF.fit(X_train_std, y_train_std)
        self.save_model(name, RF)

        df_name = self.test_model(name, X_test_std, y_test_std)

        return df_name

        # +++++++++++++++++ XGB Regressor+++++++++++++++++++++++++++++++

    def train_and_predict_ml_XGB(self, X_train_std, X_test_std, y_train_std, y_test_std, no_of_unkn, no_of_samples=20):

        name = 'XGB'

        # X_train_std, X_test_std, y_train_std, y_test_std = self.split_scale_data(X, y, seed=100, test_size=0.3)

        xgb = XGBRegressor(n_estimators=500, max_depth=7, eta=0.01, subsample=0.7, colsample_bytree=1, random_state=12)
        xgb.fit(X_train_std, y_train_std)
        self.save_model(name, xgb)

        df_name = self.test_model(name, X_test_std, y_test_std)

        return df_name

        # +++++++++++++++++ Neural Network r+++++++++++++++++++++++++++++++

    def train_and_predict_ml_NN(self, X_train_std, X_test_std, y_train_std, y_test_std, no_of_unkn, no_of_samples=20):

        import sklearn
        import tensorflow as tf
        from keras.layers import LeakyReLU
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(gpus[0], [
                    tf.config.experimental.VirtualDeviceConfiguration(memory_limit=3000)])
            except RuntimeError as e:
                print(e)

        name = 'NN'

        input_dim = X_train_std.shape[1]
        outputs = y_train_std.shape[1]
        activation = 'relu'
        num_hidden = 200

        M_in = tf.keras.layers.Input(shape=X_train_std.shape[1:], name='X_in')
        M = tf.keras.layers.Dense(num_hidden, activation=activation)(M_in)
        M = tf.keras.layers.Dropout(0.3)(M)
        M = tf.keras.layers.Dense(100, activation=activation)(M)
        M = tf.keras.layers.Dropout(0.3)(M)
        M = tf.keras.layers.Dense(70, activation=activation)(M)
        M = tf.keras.layers.Dropout(0.3)(M)
        M = tf.keras.layers.Dense(30, activation=activation)(M)
        M = tf.keras.layers.Dropout(0.3)(M)
        # M = tf.keras.layers.Dense(35, activation=activation)(M)
        # M = tf.keras.layers.Dropout(0.5)(M)
        M = tf.keras.layers.Dense(10, activation=activation)(M)
        M = tf.keras.layers.Dense(outputs)(M)
        model = tf.keras.models.Model(inputs=[M_in], outputs=M)

        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(), metrics=['mse'])
        print(model.summary())

        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                                                          patience=self.patience, restore_best_weights=True)

        # Fitting the ANN to the Training set
        history = model.fit(X_train_std, y_train_std, validation_data=(X_test_std, y_test_std),
                            epochs=self.num_epochs, batch_size=self.batch_size,
                            callbacks=[early_stopping])

        # evaluate the model
        _, train_acc = model.evaluate(X_train_std, y_train_std, verbose=0)
        _, test_acc = model.evaluate(X_test_std, y_test_std, verbose=0)
        print('Train: %.3f, test: %.3f' % (train_acc, test_acc))

        self.save_model(name, model)

        file_path = './data/NN_plots/ml_NN_{}_{}/'.format(args.network_type, args.no_of_unkn)
        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        save_path = os.path.join(file_path, '{}_train_loss.pdf'.format(args.name_of_structure))

        plt.figure(figsize=(12, 8))
        plt.plot(history.history['loss'], color='b', label="Training loss")
        plt.plot(history.history['val_loss'], color='r', label="validation loss")
        plt.legend(loc='best', shadow=True)
        plt.title('train vs val loss' + " " + f' {args.network_type} - {args.name_of_structure} - {args.no_of_unkn}')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.savefig(save_path, format="pdf")
        plt.show()

        df_name = self.test_model(name, X_test_std, y_test_std)

        return df_name

    def train_and_predict_ml_GCN(self, data, no_of_unkn, no_of_samples):
        network_type = args.network_type
        epochs = 120
        batch_size = 20 #20
        patience = 20
        learning_rate = 0.0001  # 1e-8 # 0.01

        # if network_type == 'LC_Osc':
        #     if no_of_unkn == 1:
        #         epochs = 100
        #         batch_size = 20
        #         patience = 30
        #         learning_rate = 0.001# 1e-8 # 0.01
        #
        #     elif no_of_unkn == 2:
        #         epochs = 70
        #         batch_size = 32
        #         patience = 30
        #         learning_rate = 0.0001# 1e-8 # 0.01
        #
        # elif network_type == 'CCM':
        #     if no_of_unkn == 1:
        #         epochs = 50 # 150
        #         batch_size = 20 # 32
        #         patience = 30
        #         learning_rate = 0.001# 1e-8 # 0.01
        #
        #     elif no_of_unkn == 2:
        #         epochs = 100
        #         batch_size = 100
        #         patience = 30
        #         learning_rate = 0.0001  # 1e-8 # 0.01

        loss_fn = tf.keras.losses.MeanSquaredError()

        np.random.shuffle(data)
        split = int(0.8 * len(data))
        data_tr, data_te = data[:split], data[split:]

        loader_tr = DisjointLoader(data_tr, batch_size=batch_size, epochs=epochs)
        loader_te = DisjointLoader(data_te, batch_size=batch_size)

        model = RegNet()
        opt = Adam(learning_rate=learning_rate)

        # model.compile(optimizer=opt, loss='mse', metrics=['mse'])
        if no_of_unkn == 1:
            model.compile(optimizer=opt, loss=loss_fn, metrics=tf.keras.metrics.MeanSquaredError())

        elif no_of_unkn == 2:
            model.compile(optimizer=opt,
                          loss={'output_1': loss_fn, 'output_2': loss_fn},
                          metrics={'output_1': tf.keras.metrics.MeanSquaredError(),
                                   'output_2': tf.keras.metrics.MeanSquaredError()})
        elif no_of_unkn == 3:
            model.compile(optimizer=opt,
                          loss={'output_1': loss_fn, 'output_2': loss_fn, 'output_3': loss_fn},
                          metrics={'output_1': tf.keras.metrics.MeanSquaredError(),
                                   'output_2': tf.keras.metrics.MeanSquaredError(),
                                   'output_3': tf.keras.metrics.MeanSquaredError()})

        elif no_of_unkn == 4:
            model.compile(optimizer=opt,
                          loss={'output_1': loss_fn, 'output_2': loss_fn, 'output_3': loss_fn, 'output_4': loss_fn},
                          metrics={'output_1': tf.keras.metrics.MeanSquaredError(),
                                   'output_2': tf.keras.metrics.MeanSquaredError(),
                                   'output_3': tf.keras.metrics.MeanSquaredError(),
                                   'output_4': tf.keras.metrics.MeanSquaredError()})

        file_path = './data/model/'
        if not os.path.isdir(file_path):
            os.makedirs(file_path)

        filename = os.path.join(file_path, 'ml_{}_{}_{}_{}'.format('GCN', args.network_type,
                                                                   args.name_of_structure,
                                                                   args.no_of_unkn))

        try:
            model.fit(loader_tr.load(),
                      steps_per_epoch=loader_tr.steps_per_epoch,
                      validation_data=loader_te.load(),
                      validation_steps=loader_te.steps_per_epoch,
                      epochs=epochs,
                      callbacks=[EarlyStopping(patience=patience, restore_best_weights=True)])
            model.save(filename)
            print('saved')

        except KeyboardInterrupt:
            model.save('filename')
            print('interrupted and saved')

        # load model and pred / error rate
        df = self.gcn_test(loader_te, filename)
        df_name = self.clean_dataframe(df)
        # print(df_name)
        self.save_csv_file(df_name, 'GCN')

        return df_name
        # return

    def acc_fn(self, pred, targets):
        per_err = abs((pred - targets) / targets)
        return per_err

    def add_element(self, dictn, key, value):
        if key not in dictn:
            dictn[key] = []
        dictn[key].append(value)
        return

    #
    # #############################################################################
    # Evaluate model
    # #############################################################################

    def gcn_test(self, test_loader, model_save_path):
        print("Testing model")
        print("=================")

        nw_tp = args.network_type
        no_of_samples = args.no_of_samples
        no_of_unkn = args.no_of_unkn

        step = 0
        results_te, targets_te, predicts_te = [], [], []
        result_dict = dict()
        start = time.time()
        model = tf.keras.models.load_model(model_save_path)

        for batch_eval in test_loader:
            step += 1
            inputs_eval, target_eval = batch_eval
            predictions = model(inputs_eval, training=False)

            predicts_te.append(np.array(predictions))
            # predicts_te.append(predictions)
            targets_te.append(np.array(target_eval))
            # print('1 batch done')
            if step == test_loader.steps_per_epoch:
                break

        end = time.time()
        total, agree = len(targets_te), 0

        print(predicts_te[0], targets_te[0])
        print('length of prediction:', len(predicts_te))

        print(f'\n------Evaluation results------')

        # for i in range(no_of_samples):
        for i in range(total):

            # print(predicts_te[i][0])

            if no_of_unkn == 2:

                pred = predicts_te[i][0]

                pred[0] = 10 ** (pred[0])
                targets_te[i][0][0] = 10 ** (targets_te[i][0][0])

                pred[1] = 10 ** (pred[1])
                targets_te[i][0][1] = 10 ** (targets_te[i][0][1])

                per_err_0 = (abs(pred[0] - targets_te[i][0][0]) / targets_te[i][0][0]) * 100
                per_err_1 = (abs(pred[1] - targets_te[i][0][1]) / targets_te[i][0][1]) * 100

                MSE = 0
                RMSE = 0

                self.add_element(result_dict, 'MODEL NAME', 'GCN')
                self.add_element(result_dict, 'Actual value0', targets_te[i][0][0])
                self.add_element(result_dict, 'Predicted Value0', pred[0])
                self.add_element(result_dict, 'percent_error_0', per_err_0)
                self.add_element(result_dict, 'Actual value1', targets_te[i][0][1])
                self.add_element(result_dict, 'Predicted Value1', pred[1])
                self.add_element(result_dict, 'percent_error_1', per_err_1)
                self.add_element(result_dict, 'MSE', MSE)
                self.add_element(result_dict, 'MAE', RMSE)
                self.add_element(result_dict, 'Duration', end - start)

            elif no_of_unkn == 3:

                # print(predicts_te[i])

                pred = predicts_te[i][0]
                # print('pred[0]', pred[0])

                pred[0][0] = 10 ** (pred[0][0])
                targets_te[i][0][0] = 10 ** (targets_te[i][0][0])

                pred[1][0] = 10 ** (pred[1][0])
                targets_te[i][0][1] = 10 ** (targets_te[i][0][1])

                pred[2][0] = 10 ** (pred[2][0])
                targets_te[i][0][2] = 10 ** (targets_te[i][0][2])

                per_err_0 = (abs(pred[0][0] - targets_te[i][0][0]) / targets_te[i][0][0]) * 100
                per_err_1 = (abs(pred[1][0] - targets_te[i][0][1]) / targets_te[i][0][1]) * 100
                per_err_2 = (abs(pred[2][0] - targets_te[i][0][2]) / targets_te[i][0][2]) * 100

                MSE = 0
                RMSE = 0

                self.add_element(result_dict, 'MODEL NAME', 'GCN')
                self.add_element(result_dict, 'Actual value0', targets_te[i][0][0])
                self.add_element(result_dict, 'Predicted Value0', pred[0][0])
                self.add_element(result_dict, 'percent_error_0', per_err_0)
                self.add_element(result_dict, 'Actual value1', targets_te[i][0][1])
                self.add_element(result_dict, 'Predicted Value1', pred[1][0])
                self.add_element(result_dict, 'percent_error_1', per_err_1)
                self.add_element(result_dict, 'Actual value2', targets_te[i][0][2])
                self.add_element(result_dict, 'Predicted Value2', pred[2][0])
                self.add_element(result_dict, 'percent_error_2', per_err_2)
                self.add_element(result_dict, 'MSE', MSE)
                self.add_element(result_dict, 'MAE', RMSE)
                self.add_element(result_dict, 'Duration', end - start)

            elif no_of_unkn == 4:

                pred = predicts_te[i][0]

                pred[0][0] = 10 ** (pred[0][0])
                targets_te[i][0][0] = 10 ** (targets_te[i][0][0])
                # print(f'pred[0][0] : {pred[0][0]} targets_te[i][0][0] : {targets_te[i][0][0]}')

                pred[1][0] = 10 ** (pred[1][0])
                targets_te[i][0][1] = 10 ** (targets_te[i][0][1])

                pred[2][0] = 10 ** (pred[2][0])
                targets_te[i][0][2] = 10 ** (targets_te[i][0][2])

                pred[3][0] = 10 ** (pred[3][0])
                targets_te[i][0][3] = 10 ** (targets_te[i][0][3])
                # print(f'pred[3][0] : {pred[3][0]} targets_te[i][0][3] : {targets_te[i][0][3]}')

                per_err_0 = (abs(pred[0][0] - targets_te[i][0][0]) / targets_te[i][0][0]) * 100
                per_err_1 = (abs(pred[1][0] - targets_te[i][0][1]) / targets_te[i][0][1]) * 100
                per_err_2 = (abs(pred[2][0] - targets_te[i][0][2]) / targets_te[i][0][2]) * 100
                per_err_3 = (abs(pred[3][0] - targets_te[i][0][3]) / targets_te[i][0][3]) * 100

                MSE = 0
                RMSE = 0

                self.add_element(result_dict, 'MODEL NAME', 'GCN')
                self.add_element(result_dict, 'Actual value0', targets_te[i][0][0])
                self.add_element(result_dict, 'Predicted Value0', pred[0][0])
                self.add_element(result_dict, 'percent_error_0', per_err_0)
                self.add_element(result_dict, 'Actual value1', targets_te[i][0][1])
                self.add_element(result_dict, 'Predicted Value1', pred[1][0])
                self.add_element(result_dict, 'percent_error_1', per_err_1)
                self.add_element(result_dict, 'Actual value2', targets_te[i][0][2])
                self.add_element(result_dict, 'Predicted Value2', pred[2][0])
                self.add_element(result_dict, 'percent_error_2', per_err_2)
                self.add_element(result_dict, 'Actual value3', targets_te[i][0][3])
                self.add_element(result_dict, 'Predicted Value3', pred[3][0])
                self.add_element(result_dict, 'percent_error_3', per_err_3)
                self.add_element(result_dict, 'MSE', MSE)
                self.add_element(result_dict, 'MAE', RMSE)
                self.add_element(result_dict, 'Duration', end - start)

            else:
                pred[0] = 10 ** (pred[0])
                targets_te[i][0] = 10 ** (targets_te[i][0])

                # MSE = 0
                # RMSE = 0

                MSE = mean_squared_error(targets_te[i][0], pred[0], squared=False).round(2)
                RMSE = np.sqrt(mean_absolute_error(targets_te[i][0], pred[0]).round(2))
                per_err = (abs(pred[0] - targets_te[i][0]) / targets_te[i][0]) * 100

                self.add_element(result_dict, 'MODEL NAME', 'GCN')
                self.add_element(result_dict, 'Actual value', targets_te[i][0])
                self.add_element(result_dict, 'Predicted Value', pred[0])
                self.add_element(result_dict, 'percent_error', per_err)
                self.add_element(result_dict, 'MSE', MSE)
                self.add_element(result_dict, 'MAE', RMSE)
                self.add_element(result_dict, 'Duration', end - start)

            # print(f'eval: {(10 ** (pred[0] + 1))} | targ: {10 ** (targets_te[i][0] + 1)} '
            #       f'| percent error: {per_err} | MSE: {MSE} | MAE : {MAE}')
                if per_err.all() <= 20:
                    agree += 1

                if (targets_te[i] == pred).all():
                    print(
                        f'\n-----# of Agrees: {agree} | # of Total: {total} | Accuracy: {round(float(agree / total), 2)}')

        df_name = pd.DataFrame(result_dict)

        return df_name

    def clean_dataframe(self, df):
        df = df.replace(to_replace='\[', value='', regex=True)
        df = df.replace(to_replace='\]', value='', regex=True)
        df = df.replace(to_replace='\[\[', value='', regex=True)
        df = df.replace(to_replace='\]\]', value='', regex=True)
        df = df.replace(to_replace='\W+', value='', regex=True)
        # df_conv = df
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)
        df_conv = df.loc[:, df.columns != 'MODEL NAME']
        df_conv = df_conv.astype(float)
        df_conv['MODEL NAME'] = (df['MODEL NAME'])
        first_column = df_conv.pop('MODEL NAME')
        df_conv.insert(0, 'MODEL NAME', first_column)  # Insert 'MODEL NAME' as the first column
        # df_conv.head(5)
        return df_conv


class GraphConvolutionNetwork(Model):
    def __init__(self, num_targets, num_features, hidden_units=64):
        super(GraphConvolutionNetwork, self).__init__()

        self.gcn_layers = []  # List to store GCN layers

        # First GCN layer
        self.gcn_layers.append(layers.Dense(hidden_units, activation='relu'))

        # Additional GCN layers (if needed)
        # You can add more GCN layers if your graph requires deeper processing
        self.gcn_layers.append(layers.Dense(hidden_units, activation='relu'))
        self.gcn_layers.append(layers.Dense(32, activation='relu'))

        # Output layers for each target variable
        self.output_layers = [layers.Dense(1, activation='linear') for _ in range(num_targets)]

    def call(self, inputs):
        x, a, i = inputs

        # Apply all GCN layers
        for gcn_layer in self.gcn_layers:
            x = gcn_layer(x, a)

        # Separate output for each target variable
        outputs = [output_layer(x) for output_layer in self.output_layers]

        return tf.concat(outputs, axis=1)  # Concatenate the output tensors

def mean_squared_error_multi(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

class RegNet(Model):
   
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.no_of_unkn = args.no_of_unkn
        self.conv1 = GCSConv(64, activation="relu")
        self.conv2 = GCSConv(32, activation="relu")
        self.conv3 = GCSConv(16, activation="relu")
        self.conv4 = GCSConv(16, activation="relu")
        self.global_pool = GlobalAvgPool()
        if self.no_of_unkn == 1:
            self.dense = Dense(1)
        elif self.no_of_unkn == 2:
            self.dens0 = Dense(units='1', name='output_1')
            self.dens1 = Dense(units='1', name='output_2')
        elif self.no_of_unkn == 3:
            self.dens0 = Dense(units='1', name='output_1')
            self.dens1 = Dense(units='1', name='output_2')
            self.dens2 = Dense(units='1', name='output_3')
        elif self.no_of_unkn == 4:
            self.dens0 = Dense(units='1', name='output_1')
            self.dens1 = Dense(units='1', name='output_2')
            self.dens2 = Dense(units='1', name='output_3')
            self.dens3 = Dense(units='1', name='output_4')

    def call(self, inputs):
        # x, a, i = inputs
        # x = Input(shape=(8,), name='x')
        # a = Input(shape=(None,), sparse=True)
        # i = Input(shape=(), name='segment_ids_in', dtype=tf.int32)
        x, a, i = inputs
        x = self.conv1([x, a])
        x = self.conv2([x, a])
        x = self.conv3([x, a])
        x = self.conv4([x, a])
        output = self.global_pool([x, i])
        if self.no_of_unkn == 1:
            output = self.dense(output)
            return output
        elif self.no_of_unkn == 2:
            output_1 = self.dens0(output)
            output_2 = self.dens1(output)
            return output_1, output_2
        elif self.no_of_unkn == 3:
            output_1 = self.dens0(output)
            output_2 = self.dens1(output)
            output_3 = self.dens2(output)
            outputs = [output_1, output_2, output_3]
            return outputs
            # return output_1, output_2, output_3
        elif self.no_of_unkn == 4:
            output_1 = self.dens0(output)
            output_2 = self.dens1(output)
            output_3 = self.dens2(output)
            output_4 = self.dens3(output)
            outputs = [output_1, output_2, output_3, output_4]
            return outputs
            # return output_1, output_2, output_3, output_4


def main(arg):
    print(get_memory())
    mem_limit = int(400e9)
    print('memory_limit: ', mem_limit)

    bench_dir = arg.bench_dir
    data = AnaGraphDataset(bench_dir)

    df_features, df_target = data.build_dataframe(bench_dir)

    ml = Mlearning(df_features, df_target, data)

    print('success')

    #   ######################################################
    epochs = 50
    batch_size = 32
    patience = 0
    learning_rate = 0.0001
    l2_reg = 5e-4  # Regularization rate for l2


def parse_args():
    '''
    Usual pythonic way of parsing command line arguments
    :return: all command line arguments read
    '''
    args = argparse.ArgumentParser(description='Usage: python3.10 graph_learn_reg.py [bench_dir] '
                                               '[--input feature selection]'
                                               '[network type]'
                                               '[ no of unknowns]'
                                               '[no of output samples]'
                                               '[model name]')
    args.add_argument('bench_dir', help='circuit file directory')
    # args.add_argument('train_dir', help='read in train dataset')
    # args.add_argument('test_dir', help='read in test dataset')

    args.add_argument('-b', "--batch_size", default=10, type=int,
                      help="Number of samples per training batch")

    args.add_argument('-e', "--epochs", default=500, type=int,
                      help="Number of iterations the whole dataset of graphs is traversed")

    args.add_argument('-lr', "--learning_rate", default=0.001, type=float,
                      help="Learning rate to optimize the loss function")

    args.add_argument('-hl', "--hidden_layers", default=50, type=int,
                      help="number of hidden layers")

    args.add_argument('-pc', "--patience", default=20, type=int,
                      help="patience value")

    args.add_argument('-struct', "--name_of_structure", default='i_o_val', type=str,
                      help="type of features for training ==> 'i_o', 'i_o_val', 'i_o_val_topo' ")

    args.add_argument('-nt', "--network_type", default='LC_Osc', type=str,
                      help="type of network ==> 'Resistor', 'RC', 'ALPF_1ST','CCM', 'LC_Osc', 'TWG', 'ALPF_2ND' ")

    args.add_argument('-unkn', "--no_of_unkn", default=1, type=int,
                      help="number of unknowns to predict")

    args.add_argument('-ns', "--no_of_samples", default=20, type=int,
                      help="number of sample predicted values from test result")

    args.add_argument('-model', "--ml_method", default='RF', type=str,
                      help="Machine learning Model for prediction or"
                           " plotting command ==> 'LR', 'SGD','DT','RF',"
                           " 'XGB', 'NN', 'GCN', 'PLOT'")

    args.add_argument('-op', "--operation", default='train', type=str,
                      help="train or test")

    args.add_argument('-df', "--df_name", default='result', type=str,
                      help="name of dataframe to save predicted values from test result")

    args.add_argument('-sz', "--sample_size", default=1.0, type=float,
                      help="number of samples from input feature vector")

    return args.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)




