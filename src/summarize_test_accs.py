import os, glob
import pandas as pd
import numpy as np

target_dir = 'results/modeltesting'
target_folders = sorted(glob.glob(os.path.join(target_dir,'nn*')))
order = ['alpha','beta','gamma']
test_acc_dict = {}
# create keys and list of 999s
for i in target_folders:
    key = os.path.basename('_'.join(i.split('_')[:-1]))
    test_acc_dict[key] = [999,999,999]
for i in target_folders:
    mae_table = np.loadtxt(os.path.join(i,'train_val_test_mape_unscaled_scaled.txt'))
    test_mae = mae_table[-1,-1]
    mode = i.split('_')[-1]
    key = os.path.basename('_'.join(i.split('_')[:-1]))
    col_id = np.where(np.array(order) == mode)[0][0]
    test_acc_dict[key][col_id] = test_mae

# list of keys
all_keys = list(test_acc_dict.keys())
# array of values in same order as keys
all_test_maes = np.array([test_acc_dict[i] for i in all_keys])
# get best result for each mode
best_maes = np.min(all_test_maes,axis=0)
# identify best model for each mode
best_models_per_mode = {}
for i,mae in enumerate(best_maes):
    best_model = np.where(all_test_maes[:,i]==mae)[0][0]
    best_models_per_mode[order[i]]=all_keys[best_model]

for i in best_models_per_mode:
    print('%s:'%i,best_models_per_mode[i])


a = pd.DataFrame(np.hstack([np.array(all_keys).reshape(len(all_keys),1),all_test_maes]))
a.to_csv('results/modeltesting/modeltesting_acc_overview.txt',sep='\t')
