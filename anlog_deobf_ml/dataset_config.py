import os

usr_dir = os.path.expanduser('~')
proj_dir = os.path.join(usr_dir, 'PycharmProjects/pythonProject1/')


def check_dir_exists(dirs: str):
    if not os.path.exists(dirs):
        os.makedirs(dirs)
        print(f'-- make dir {dirs} ---')


train_base_path = 'gcn_ana/Benches_for_Training/'
train_dataset_base_path = os.path.join(proj_dir, train_base_path)

train_res_path_1 = 'RES_NW_1/'
check_dir_exists(train_dataset_base_path + train_res_path_1)
train_resistor_dataset_path_1 = os.path.join(train_dataset_base_path, train_res_path_1)

train_rc_path_1 = 'RC_Circuit_1/'
check_dir_exists(train_dataset_base_path + train_rc_path_1)
train_rc_dataset_path_1 = os.path.join(train_dataset_base_path, train_rc_path_1)

train_mos_path_1 = 'MOS_1/'
check_dir_exists(train_dataset_base_path + train_mos_path_1)
train_mos_dataset_path_1 = os.path.join(train_dataset_base_path, train_mos_path_1)

train_lc_path_1 = 'LC_Osc_1/'
check_dir_exists(train_dataset_base_path + train_lc_path_1)
train_lc_dataset_path_1 = os.path.join(train_dataset_base_path, train_lc_path_1)

train_twg_path_1 = 'TWG_1/'
check_dir_exists(train_dataset_base_path + train_twg_path_1)
train_twg_dataset_path_1 = os.path.join(train_dataset_base_path, train_twg_path_1)

# train_dataset_files = [train_resistor_dataset_path, train_rc_dataset_path,
#                        train_mos_dataset_path,
#                        train_lc_dataset_path, train_twg_dataset_path]
# ***************************************************************************************

train_res_path_2 = 'RES_NW_2/'
check_dir_exists(train_dataset_base_path + train_res_path_2)
train_resistor_dataset_path_2 = os.path.join(train_dataset_base_path, train_res_path_2)

train_rc_path_2 = 'RC_Circuit_2/'
check_dir_exists(train_dataset_base_path + train_rc_path_2)
train_rc_dataset_path_2 = os.path.join(train_dataset_base_path, train_rc_path_2)

train_mos_path_2 = 'MOS_2/'
check_dir_exists(train_dataset_base_path + train_mos_path_2)
train_mos_dataset_path_2 = os.path.join(train_dataset_base_path, train_mos_path_2)

train_lc_path_2 = 'LC_Osc_2/'
check_dir_exists(train_dataset_base_path + train_lc_path_2)
train_lc_dataset_path_2 = os.path.join(train_dataset_base_path, train_lc_path_2)

train_twg_path_2 = 'TWG_2/'
check_dir_exists(train_dataset_base_path + train_twg_path_2)
train_twg_dataset_path_2 = os.path.join(train_dataset_base_path, train_twg_path_2)

train_alpf_1st_path_2 = 'ALPF_1ST_2/'
check_dir_exists(train_dataset_base_path + train_alpf_1st_path_2)
train_alpf_1st_dataset_path_2 = os.path.join(train_dataset_base_path, train_alpf_1st_path_2)

train_alpf_2nd_path_2 = 'ALPF_2ND_2/'
check_dir_exists(train_dataset_base_path + train_alpf_2nd_path_2)
train_alpf_2nd_dataset_path_2 = os.path.join(train_dataset_base_path, train_alpf_2nd_path_2)

train_abpf_4th_path_2 = 'ABPF_4TH_2/'
check_dir_exists(train_dataset_base_path + train_abpf_4th_path_2)
train_abpf_4th_dataset_path_2 = os.path.join(train_dataset_base_path, train_abpf_4th_path_2)


# ***************************************************************************************

train_res_path_3 = 'RES_NW_3/'
check_dir_exists(train_dataset_base_path + train_res_path_3)
train_resistor_dataset_path_3 = os.path.join(train_dataset_base_path, train_res_path_3)

train_mos_path_3 = 'MOS_3/'
check_dir_exists(train_dataset_base_path + train_mos_path_3)
train_mos_dataset_path_3 = os.path.join(train_dataset_base_path, train_mos_path_3)

train_lc_path_3 = 'LC_Osc_3/'
check_dir_exists(train_dataset_base_path + train_lc_path_3)
train_lc_dataset_path_3 = os.path.join(train_dataset_base_path, train_lc_path_3)

train_twg_path_3 = 'TWG_3/'
check_dir_exists(train_dataset_base_path + train_twg_path_3)
train_twg_dataset_path_3 = os.path.join(train_dataset_base_path, train_twg_path_3)

train_abpf_4th_path_3 = 'ABPF_4TH_3/'
check_dir_exists(train_dataset_base_path + train_abpf_4th_path_3)
train_abpf_4th_dataset_path_3 = os.path.join(train_dataset_base_path, train_abpf_4th_path_3)

# ***************************************************************************************
train_abpf_4th_path_4 = 'ABPF_4TH_4/'
check_dir_exists(train_dataset_base_path + train_abpf_4th_path_4)
train_abpf_4th_dataset_path_4 = os.path.join(train_dataset_base_path, train_abpf_4th_path_4)

# ****************************************************************************************
train_chua_ckt = 'chua_2/'
check_dir_exists(train_dataset_base_path + train_chua_ckt)
train_chua_ckt_path_2 = os.path.join(train_dataset_base_path, train_chua_ckt)
# ****************************************************************************************
train_cnfg_ccm = 'conf_ccm_1/'
check_dir_exists(train_dataset_base_path + train_cnfg_ccm)
train_cnfg_ccm_path_1 = os.path.join(train_dataset_base_path, train_cnfg_ccm)
# ****************************************************************************************
train_ota_bpf = 'ota_bpf_6/'
check_dir_exists(train_dataset_base_path + train_ota_bpf)
train_ota_bpf_path_4 = os.path.join(train_dataset_base_path, train_ota_bpf)

# train_dataset_files = [train_resistor_dataset_path, train_rc_dataset_path,
#                        train_mos_dataset_path,
#                        train_lc_dataset_path, train_twg_dataset_path]
# ***************************************************************************************
#    TEST DATASET PATHS
# ***************************************************************************************

test_base_path = 'gcn_ana/Benches_for_Testing/'
test_dataset_base_path = os.path.join(proj_dir, test_base_path)

test_res_path = 'RES_NW/'
check_dir_exists(test_dataset_base_path + test_res_path)
test_resistor_dataset_path = os.path.join(test_dataset_base_path, test_res_path)

test_rc_path = 'RC_Circuit/'
check_dir_exists(test_dataset_base_path + test_rc_path)
test_rc_dataset_path = os.path.join(test_dataset_base_path, test_rc_path)

test_mos_path = 'MOS/'
check_dir_exists(test_dataset_base_path + test_mos_path)
test_mos_dataset_path = os.path.join(test_dataset_base_path, test_mos_path)

test_lc_path = 'LC_Osc_1/'
check_dir_exists(test_dataset_base_path + test_lc_path)
test_lc_dataset_path_1 = os.path.join(test_dataset_base_path, test_lc_path)

test_twg_path = 'TWG/'
check_dir_exists(test_dataset_base_path + test_twg_path)
test_twg_dataset_path = os.path.join(test_dataset_base_path, test_twg_path)

# test_dataset_files = [test_resistor_dataset_path, test_rc_dataset_path,
#                       test_mos_dataset_path,
#                       test_lc_dataset_path, test_twg_dataset_path]
