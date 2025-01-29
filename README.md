# Towards Machine-Learning-based Oracle-Guided Analog Circuit Deobfuscation


Code for generic analog circuit deobfuscation presented at the IEEE International Test Conference (ITC) 2024 conference.
Link for the paper: https://ieeexplore.ieee.org/abstract/document/10766677


## Overview
This repository contains the implementation of an oracle-guided deobfuscation framework for analog circuits using machine learning techniques. The framework automates the process of recovering obfuscated parameters in analog circuits by training models on synthetic data generated through circuit simulations.


## Features
1. Machine learning models to predict unknown circuit parameters
2. Supports multiple circuit types including resistive networks, op-amp circuits, and LC oscillators
3. Utilizes uncertainty sampling for efficient querying
4. Implements Graph Neural Networks (GNN) for topology-aware learning
5. Compatible with open-source circuit simulators for data generation


## Requirements
### Software Dependencies

To run this project, you need the following libraries:
1. Python 3.10+
2. Scikit-learn
3. PySpice
4. NgSpice
5. Spektral (for GNN models)
6. NumPy
7. Pandas
8. Matplotlib
   

### Hardware Requirements
The implementation is optimized for a system with:
Multi-core CPU (e.g., AMD EPYC or Intel Xeon recommended)
128GB RAM (for large dataset training)
GPU (optional, for accelerating neural network training)


## Installation
Clone the repository:
```
git clone https://github.com/dipaishaj/anlog_deobf_ml.git
cd anlog_deobf_ml
```


## Create a virtual environment and install dependencies:
```
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate
pip install -r requirements.txt
```


## Usage
### 1. Data Generation
Run the circuit simulation to generate training data:

```
python3 dataset_generate.py --type_of_circuit 'lpf' --noise_input 0 --process_variation 1 --pv_tolerance 10 --no_of_unkn 2 --size_of_dataset 1000 --type_of_osc 'rc'
--type_of_filter 'lpf_1' --simulation_method 'ac'`
```

1. '--type_of_circuit', default='osc', type=str,'Type of a circuit -> osc, filter'
2. '--noise_input', default=0, type=int,'Add Noise in the circuit -> 0: no noise or 1: noise'
3. '--process_variation', default=0, type=int,' Add process variation in the circuit -> 0: no variation, 1: variation'
4. '--pv_tolerance', default=1, type=int,'Add % of tolerance for process variation -> between 0 to 20'
5. '--no_of_unkn', default=1, type=int,'Number of Unknown components -> between 1 to 4'
6. '--size_of_dataset', default=500, type=int, help='Number of benchmark circuits'
7. '--type_of_osc', default='wb', type=str, 'Type of Oscillator circuit -> 1: wein bridge, 2: RC phase shift'
8. '--type_of_filter', default='lpf_1', type=str,'Type of Filter circuit -> 1: LPF 1st order, 2: LPF 2nd order'
9. '--simulation_method', default='transient', type=str,'type of simulation method -> transient, operating_point, ac'



### 2. Model Training
Train a machine learning model for circuit deobfuscation:

```
python3 GCN_Learn_Analog_reg.py ./training_data/# --name_of_structure 'i_o_val_topo' --network_type 'TWG' --no_of_unkn 2 --ml_method 'RF' -operation 'train'
```

1. "bench_dir", 'circuit file directory'
2. "--batch_size", default=10, type=int, "Number of samples per training batch"
3. "--epochs", default=500, type=int, "Number of iterations the whole dataset of graphs is traversed"
4. "--learning_rate", default=0.001, type=float, "Learning rate to optimize the loss function"
5. "--hidden_layers", default=50, type=int,"number of hidden layers"
6. "--patience", default=20, type=int,"patience value"
7. "--name_of_structure", default='i_o_val', type=str,"type of features for training ==> 'i_o', 'i_o_val', 'i_o_val_topo' 
8. "--network_type", default='LC_Osc', type=str, "type of network ==> 'Resistor', 'RC', 'ALPF_1ST','CCM', 'LC_Osc', 'TWG', 'ALPF_2ND' 
9. "--no_of_unkn", default=1, type=int,"number of unknowns to predict"
10. "--no_of_samples", default=20, type=int, "number of samples predicted values from test result"
11. "--ml_method", default='RF', type=str, "Machine learning Model for prediction or plotting command" ==> 'LR', 'SGD', 'DT', 'RF', 'XGB', 'NN', 'GCN', 'PLOT'"
12. "--operation", default='train', type=str,"train or test"
13. "--df_name", default='result', type=str, "name of the data frame to save predicted values from test result"
14. "--sample_size", default=1.0, type=float,"number of samples from input feature vector"


### 3. Prediction
Use a trained model to predict unknown circuit parameters:

```
python3 GCN_Learn_Analog_reg.py ./test_data/# --name_of_structure 'i_o_val_topo' --network_type 'TWG' --no_of_unkn 2 --ml_method 'RF' -operation 'test'
```


## Citation
If you use this code for research, please cite:

- D. Jain, G. Zhao, R. Datta, and K. Shamsi, "Towards Machine-Learning-based Oracle-Guided Analog Circuit Deobfuscation," 2024 IEEE International Test Conference (ITC), San Diego, CA, USA, 2024, pp. 323-332, doi: 10.1109/ITC51657.2024.00053.
  
- @INPROCEEDINGS{10766677,
  author={Jain, Dipali and Zhao, Guangwei and Datta, Rajesh and Shamsi, Kaveh},
  booktitle={2024 IEEE International Test Conference (ITC)}, 
  title={Towards Machine-Learning-based Oracle-Guided Analog Circuit Deobfuscation}, 
  year={2024},
  volume={},
  number={},
  pages={323-332},
  keywords={Uncertainty;Hardware security;Reverse engineering;Machine learning;Manuals;Analog circuits;Predictive models;Mathematical models;Graph neural      
  networks;Synthetic data;Analog Circuits;Deobfuscation;Circuit Learning;Machine Learning;Graph Neural Networks;Uncertainty sampling},
  doi={10.1109/ITC51657.2024.00053}}


## Contact
For questions or collaboration, please contact: dipali.jain@utdallas.edu


