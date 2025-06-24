'''Import API & Library'''
import warnings
warnings.simplefilter("ignore", UserWarning)
import argparse

import math
from torchsummary import summary as summary

import torch # For building the networks 
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.nets import *

from utils import *

from adamp import AdamP

from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from medcam import medcam
from medcam import *

from skimage.transform import resize
from scipy import ndimage
import sys 
import argparse
import pdb
import pandas as pd
import os
from os.path import join
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import models
from torchvision import datasets ,transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import sklearn
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import psutil
import random
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from dataset import EditSurvDataset, EditProcSurvDataset, Eval_dataset,sample_minibatch_for_global_loss_opti_Subject_3D, Image_N_TabularDatasetFP, Image_N_TabularDataset
import torch.nn as nn
import warnings
warnings.filterwarnings('ignore') 
import os
import pdb
import wandb
from datetime import datetime
from sksurv.metrics import concordance_index_ipcw, brier_score
from sklearn.preprocessing import StandardScaler

# for DINO
import argparse
import logging
import os
import sys
import numpy as np


import torch

import vision_transformer_3d as vits_3d
from load_VIT import *


from datetime import datetime
now = datetime.now()



def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
    
    parser.add_argument('--data_dir', type=str, default='/mnt/hdd1/chlee/data')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_gpu_id', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=65) 
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
    parser.add_argument('--spec_duration', type=str, default='OS') # 'OS' # '1yr'
    parser.add_argument('--spec_event', type=str, default='death') # 'death' # 'prog'
    parser.add_argument('--dataset_name', type=str, default='SNUH_severance') # 'TCGA' # 
    parser.add_argument('--dataset_list', nargs='+', default=['SNUH', 'severance'], help='selected_training_datasets') # ['UCSF','UPenn','TCGA','SNUH']]
    parser.add_argument('--remove_idh_mut', default=False, type=str2bool)
    parser.add_argument('--save_grad_cam', default=False, type=str2bool)
    parser.add_argument('--biopsy_exclusion', default=False, type=str2bool)
    parser.add_argument('--num_category', type=int, default=201)
    parser.add_argument('--net_architect', default='Tabular', type=str) # 'SEResNext50'
    parser.add_argument('--cpu_start', '-c', type=int, default=0, help="bootstrapping number")
    # parser.add_argument('--FP', action='store_true', help='use npy embedding')
    # parser.add_argument('--finetuning', action='store_true', help='encoder finetuning')
    parser.add_argument('--cohort', type=str, choices=['all_data_embedding', 'korean_data_embedding', 'TCGA_data_embedding', 'UPennTCGA_data_embedding', 'UPenn_data_embedding'], default='UPennTCGA_data_embedding', help="cohort of operation: 'all_data_embedding' or 'korean_data_embedding' or 'TCGA_data_embedding' or 'UPennTCGA_data_embedding' or 'UPenn_data_embedding'")
    parser.add_argument('--wandb_off', action='store_true', help='wandb_off/on')
    parser.add_argument('--drop_rate', type=float, default=0.3, help="dropout rate: [0,1]")

    parser.add_argument('--tabular_layer_depth', type=int, default=0, help="-1: None(nn.Identity())/ 0: linear / 1~: non-linear")
    parser.add_argument('--tabular_exdim', type=int, default=4, help="in tabular layer's expand dimension ratio (e.g. hidden_dim = 64 * tabular_exdim = 2 -> 128)")
    parser.add_argument('--tabular_layer_out_dim', type=int, default=64)

    parser.add_argument('--hazard_layer_depth', type=int, default=0, help="0: linear / 1~: non-linear")
    parser.add_argument('--hazard_exdim', type=int, default=4, help="in hazard layer's expand dimension ratio (e.g. hidden_dim = 64 * hazard_exdim = 2 -> 128)")
    parser.add_argument('--gbm_only', action='store_true', help='grade 4 GBM only')
    
    parser.add_argument('--eps', type=float, default=1e-8)
    return parser


#%%
main_args = get_args_parser().parse_args()

main_args.compart_name = 'resized'
main_args.experiment_mode='none_partition'
main_args.scheduler = 'CosineAnnealingLR'
main_args.T_max = 10
main_args.T_0 = 10
main_args.min_lr = 1e-6
main_args.weight_decay = 1e-5
main_args.smoothing = 0.3
# self.batch_size = 183 # CNNs : 64, Transformers : 16
main_args.sequence = ['t1','t2','t1ce','flair']
main_args.data_key_list = [
    'sex',
    'age',
    'gbl',
    'idh',
    'mgmt',
    'eor',
    'kps',
    'who_grade'
    ]

if main_args.gbm_only:
    eval_times = [3, 6, 12, 18, 24]
else:
    eval_times = [3,6,12,24,48]
    
p = psutil.Process()
p.cpu_affinity(range(main_args.cpu_start, main_args.cpu_start+4))

seed =main_args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
sklearn.random.seed(seed)

now = datetime.now()
if not main_args.wandb_off:
    WANDB_AUTH_KEY = 'your_wandb_auth_key_here'  # Replace with your actual WandB key

    wandb.login(key=WANDB_AUTH_KEY)
    if main_args.gbm_only:
        wandb.init(project="mri_",
                name=f"tabular_downstream_{main_args.seed}_GBM_only",
                notes=f"{now.strftime('%Y-%m-%d_%H:%M:%S')}",
                tags = ["nonTCGA2korean","gbm_only"]
                )
    else:
        wandb.init(project="mri_",
                name=f"tabular_downstream_{main_args.seed}",
                notes=f"{now.strftime('%Y-%m-%d_%H:%M:%S')}",
                tags = ["nonTCGA2korean"]
                )                

    print(f'args:{main_args.__dict__}')
    wandb.config.update(main_args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# gpu_index = 4
# torch.cuda.set_device(gpu_index)  # GPU 번호 설정
PATH = '/mnt/hdd1/chlee/code/check/tabular/'
os.makedirs(PATH, exist_ok=True) 

combine_img(main_args)

main_args.device=device



cohort = "_".join(main_args.dataset_list)
if main_args.gbm_only:
    df = pd.read_csv(f'/mnt/hdd1/chlee/data/{cohort}_total_OS_GBM_v3_filtered.csv')
    
    train_valid_df, test_df = train_test_split(df, 
                                          test_size=0.2, 
                                          random_state=main_args.seed)

    train_df, valid_df = train_test_split(train_valid_df,
                                      test_size=0.25,
                                      random_state=main_args.seed)
else:
    df = pd.read_csv(f'/mnt/hdd1/chlee/data/{cohort}_total_OS_all_v3_filtered.csv')
    train_valid_df, test_df = train_test_split(df, 
                                           test_size=0.2, 
                                           random_state=main_args.seed)

    train_df, valid_df = train_test_split(train_valid_df,
                                      test_size=0.25,
                                      random_state=main_args.seed)

train_ids = train_df['id'].values
valid_ids = valid_df['id'].values 
test_ids = test_df['id'].values

os.makedirs(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)

# Save IDs
np.save(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/train_ids.npy', train_ids)
np.save(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/valid_ids.npy', valid_ids)
np.save(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/test_ids.npy', test_ids)

scaler = StandardScaler()
age_train_scaled = scaler.fit(train_df[['age']])

train_dataset = Image_N_TabularDatasetFP(train_df, main_args, scaler=age_train_scaled)
if main_args.dataset_list != 'TCGA_UCSF_UPenn':
    train_dataloader = DataLoader(train_dataset, batch_size=main_args.batch_size, shuffle=True, num_workers=4, drop_last=False)
else:
    train_dataloader = DataLoader(train_dataset, batch_size=main_args.batch_size, shuffle=True, num_workers=4, drop_last=True)

train_eval_dataset = Eval_dataset(train_df, main_args, dataset_name=None)
train_eval_dataloader = DataLoader(train_eval_dataset, batch_size=main_args.batch_size, shuffle=False, num_workers=4, drop_last=False)

valid_dataset = Image_N_TabularDatasetFP(valid_df, main_args, scaler=age_train_scaled)
valid_dataloader = DataLoader(valid_dataset, batch_size=main_args.batch_size, shuffle=False, num_workers=4, drop_last=False)

test_dataset = Image_N_TabularDatasetFP(test_df, main_args, scaler=age_train_scaled)
test_dataloader = DataLoader(test_dataset, batch_size=main_args.batch_size, shuffle=False, num_workers=4, drop_last=False)

external_df = pd.read_csv(f'/mnt/hdd1/chlee/data/UPenn_total_OS_all_v3_filtered.csv')
external_dataset = Image_N_TabularDatasetFP(external_df, main_args, scaler=age_train_scaled)
external_dataloader = DataLoader(external_dataset, batch_size=main_args.batch_size, shuffle=False, num_workers=4, drop_last=False)

# test_gbm_df = test_df[test_df['who_grade'] == 4.]
# test_gbm_dataset = Image_N_TabularDatasetFP(test_gbm_df, main_args, scaler=age_train_scaled)
# test_gbm_dataloader = DataLoader(test_gbm_dataset, batch_size=main_args.batch_size, shuffle=False, num_workers=4, drop_last=False)

if main_args.tabular_layer_depth == -1:
    tabular_layer = nn.Identity()
elif main_args.tabular_layer_depth == 0:
    tabular_layer = nn.Linear(len(main_args.data_key_list), main_args.tabular_layer_out_dim)
else:
    tabular_layer = MutilLayer_Outdim(input_dim= len(main_args.data_key_list), hidden_dim=main_args.tabular_layer_out_dim*main_args.tabular_exdim, out_dim=main_args.tabular_layer_out_dim , n_layers=main_args.tabular_layer_depth , drop_rate=main_args.drop_rate)



model = Tabular_HazardNet(main_args, tabular_layer, main_args.tabular_layer_out_dim).to(device)


'''Optimizer, Loss Function'''

optimizer = AdamP(model.parameters(), lr=main_args.lr, weight_decay=main_args.weight_decay)

scheduler = fetch_scheduler(optimizer, main_args)

dataloaders = {
    'train' : train_dataloader,
    'valid' : valid_dataloader
}

class SetPartition:
    img_size_x = 120
    num_channels = 80

partition_cfg = SetPartition()

''' Training Function '''

start = time.time()
best_loss = 100
history = defaultdict(list)
model = model.to(device)

internal_test_predictions = {}

# Training loop
for epoch in tqdm(range(1, main_args.epochs+1)):
    model.train()
    cost = 0.0
    
    for i, (ids, x_mb, _, t_mb, k_mb, m1_mb, m2_mb, m3_mb) in enumerate(train_dataloader):
        x_mb = x_mb.to(device, dtype=torch.float)
        t_mb = t_mb.to(device)
        k_mb = k_mb.to(device)
        m1_mb = m1_mb.to(device)
        m2_mb = m2_mb.to(device)
        m3_mb = m3_mb.to(device)

        out = model(x_mb)
        loss = nll_loss(out, m1_mb, m2_mb, k_mb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        cost += loss

    cost = cost / len(train_dataloader)
    cost_value = cost.detach().cpu().numpy()
    
    if not main_args.wandb_off:
        wandb.log({'Train Loss': cost_value}, step=epoch)
    print(f'Train_loss: ', cost_value)

    # Validation step

    model.eval()
    with torch.no_grad():
        cost = 0.0
        T_tr=[]
        E_tr=[]
        T_vl=[]
        E_vl=[]
        
        S_list=[]
        W_list=[]
        for i, (t_mb, k_mb) in enumerate(train_eval_dataloader):
            T_tr.append(t_mb)
            E_tr.append(k_mb)
        T_tr=torch.cat(T_tr,dim=0)
        E_tr=torch.cat(E_tr,dim=0)
        
        T_tr = T_tr.cpu()
        E_tr = E_tr.cpu()
        for i, (ids, valid_data, _, t_mb, k_mb, m1_mb, m2_mb, m3_mb) in enumerate(valid_dataloader):
            # pdb.set_trace()
            X_val = valid_data.to(device, dtype=torch.float)
            t_mb = t_mb.to(device)
            k_mb = k_mb.to(device)
            m1_mb = m1_mb.to(device)
            m2_mb = m2_mb.to(device)
            m3_mb = m3_mb.to(device)
            T_vl.append(t_mb)
            E_vl.append(k_mb)

            out = model(X_val)
            log_surv_out = torch.log(1. - out + main_args.eps)
            H = log_surv_out.cumsum(1)
            S = torch.exp(H)
            W = 1-S
            loss=nll_loss(out ,m1_mb ,m2_mb ,k_mb)
            cost += loss

            S_list.append(S)
            W_list.append(W)
        T_vl=torch.cat(T_vl,dim=0)
        E_vl=torch.cat(E_vl,dim=0)
        
        S_list=torch.cat(S_list,dim=0)
        W_list=torch.cat(W_list,dim=0)

        T_vl = T_vl.cpu()
        E_vl = E_vl.cpu()
        S_list = S_list.cpu()
        W_list = W_list.cpu()



        cost = cost / len(valid_dataloader)
        cost_value=cost.detach().cpu().numpy()
        if not main_args.wandb_off:
            wandb.log({f'Valid Loss': cost_value}, step=(epoch)) 
        print(f'Valid Loss: ',cost_value)

        tr_y_structured =  [(np.asarray(E_tr)[i], np.asarray(T_tr)[i]) for i in range(len(E_tr))]
        tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])

        vl_y_structured =  [(np.asarray(E_vl)[i], np.asarray(T_vl)[i]) for i in range(len(E_vl))]
        vl_y_structured = np.array(vl_y_structured, dtype=[('status', 'bool'),('time','<f8')])    

        # to compute brier-score without error
        T_vl_ = pd.DataFrame(T_vl, columns=['duration_'+main_args.spec_event])
        T_tr_ = pd.DataFrame(T_tr, columns=['duration_'+main_args.spec_event])

        T_vl_.loc[T_vl_['duration_'+main_args.spec_event] > T_tr_['duration_'+main_args.spec_event].max()] = T_tr_['duration_'+main_args.spec_event].max()
        vl_y_structured_ =  [(np.asarray(E_vl)[i], np.asarray(T_vl_)[i]) for i in range(len(E_vl))]
        vl_y_structured_ = np.array(vl_y_structured_, dtype=[('status', 'bool'),('time','<f8')])

        RESULTS1 = np.zeros([len(eval_times)])
        RESULTS2 = np.zeros([len(eval_times)])

        for t, eval_time in enumerate(eval_times):
        
            RESULTS1[t] = concordance_index_ipcw(tr_y_structured, vl_y_structured, W_list[:, eval_time], tau=eval_time)[0]
            RESULTS2[t] = brier_score(tr_y_structured, vl_y_structured_, S_list[:, eval_time], times=eval_time)[1][0]


    

        
        C_Index = np.round(RESULTS1, 4)
        Brier_Score = np.round(RESULTS2, 4)
        if not main_args.wandb_off:
            for i in range (len(C_Index)):
                wandb.log({f'Valid: C_Index_{eval_times[i]}': C_Index[i]}, step=(epoch))
            for i in range (len(Brier_Score)):
                wandb.log({f'Valid: Brier_Score_{eval_times[i]}': Brier_Score[i]}, step=(epoch))
            
            wandb.log({f'Valid: C_Index_mean': C_Index.mean()}, step=(epoch))
            wandb.log({f'Valid: Brier_Score_mean': Brier_Score.mean()}, step=(epoch))
        print(f'Valid C_Index_mean: ',C_Index.mean())
        print(f'Valid Brier_Score_mean: ',Brier_Score.mean())

        os.makedirs(f'./valid_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)
        f = open(f'./valid_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/internal_downstream_seed_{main_args.seed}.txt', 'a')
        f.write('epoch {}'.format(epoch) + '\n')
        f.write(str(C_Index) + '\n')
        f.write('*' + str(C_Index.mean()) + '\n')
        
        f.write(str(Brier_Score) + '\n')
        f.write('*' + str(Brier_Score.mean()) + '\n\n')
        
        if epoch == 1:
            threshold = C_Index.mean()
            best_model = model.state_dict()
            ee = epoch
        else:
            if threshold < C_Index.mean():
                print(threshold, C_Index.mean())
                best_model = model.state_dict()
                ee = epoch
                threshold = C_Index.mean()
                print(epoch)
            
    os.makedirs(f'./model/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)
    model_path = f'./model/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}_downstream_checkpoint.pth'
    torch.save(best_model, model_path)

# Load best model for final evaluation
model.load_state_dict(torch.load(model_path))

##test_step
model.eval()
with torch.no_grad():
    cost = 0.0
    T_tr=[]
    E_tr=[]
    T_ts=[]
    E_ts=[]
    
    S_list=[]
    W_list=[]
    for i, (t_mb, k_mb) in enumerate(train_eval_dataloader):
        T_tr.append(t_mb)
        E_tr.append(k_mb)
    T_tr=torch.cat(T_tr,dim=0)
    E_tr=torch.cat(E_tr,dim=0)
    
    T_tr = T_tr.cpu()
    E_tr = E_tr.cpu()
    for i, (ids, Test_data, _, t_mb, k_mb, m1_mb, m2_mb, m3_mb) in enumerate(test_dataloader):
        X_ts = Test_data.to(device, dtype=torch.float)
        t_mb = t_mb.to(device)
        k_mb = k_mb.to(device)
        m1_mb = m1_mb.to(device)
        m2_mb = m2_mb.to(device)
        m3_mb = m3_mb.to(device)
        T_ts.append(t_mb)
        E_ts.append(k_mb)

        out = model(X_ts)
        log_surv_out = torch.log(1. - out + main_args.eps)
        H = log_surv_out.cumsum(1)
        S = torch.exp(H)
        W = 1-S
        # proc_label = proc_label.to(device)
        loss=nll_loss(out ,m1_mb ,m2_mb ,k_mb)
        cost += loss

        S_list.append(S)
        W_list.append(W)

        # 각 환자별로 데이터 저장
        for idx in range(len(ids)):
            internal_test_predictions[ids[idx]] = {
                'true_time': t_mb[idx].cpu(),
                'true_event': k_mb[idx].cpu(),
                'survival_probs': S[idx].cpu(),  # shape: (n_eval_times,)
                'hazard_probs': W[idx].cpu(),  # shape: (n_eval_times,)
                'eval_times': eval_times
            }
    T_ts=torch.cat(T_ts,dim=0)
    E_ts=torch.cat(E_ts,dim=0)
    
    S_list=torch.cat(S_list,dim=0)
    W_list=torch.cat(W_list,dim=0)

    T_ts = T_ts.cpu()
    E_ts = E_ts.cpu()
    S_list = S_list.cpu()
    W_list = W_list.cpu()



    cost = cost / len(test_dataloader)
    cost_value=cost.detach().cpu().numpy()

    if not main_args.wandb_off:
        wandb.log({f'Test Loss': cost_value}, step=(epoch)) 
    print(f'Test: ',cost_value)

    tr_y_structured =  [(np.asarray(E_tr)[i], np.asarray(T_tr)[i]) for i in range(len(E_tr))]
    tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])

    ts_y_structured =  [(np.asarray(E_ts)[i], np.asarray(T_ts)[i]) for i in range(len(E_ts))]
    ts_y_structured = np.array(ts_y_structured, dtype=[('status', 'bool'),('time','<f8')])    

    # to compute brier-score without error
    T_ts_ = pd.DataFrame(T_ts, columns=['duration_'+main_args.spec_event])
    T_tr_ = pd.DataFrame(T_tr, columns=['duration_'+main_args.spec_event])

    internal_test_predictions['tr_y_structured'] = tr_y_structured
    internal_test_predictions['T_tr_'] = T_tr_

    T_ts_.loc[T_ts_['duration_'+main_args.spec_event] > T_tr_['duration_'+main_args.spec_event].max()] = T_tr_['duration_'+main_args.spec_event].max()
    ts_y_structured_ =  [(np.asarray(E_ts)[i], np.asarray(T_ts_)[i]) for i in range(len(E_ts))]
    ts_y_structured_ = np.array(ts_y_structured_, dtype=[('status', 'bool'),('time','<f8')])

    RESULTS1 = np.zeros([len(eval_times)])
    RESULTS2 = np.zeros([len(eval_times)])
    
    for t, eval_time in enumerate(eval_times):
    
        RESULTS1[t] = concordance_index_ipcw(tr_y_structured, ts_y_structured, W_list[:, eval_time], tau=eval_time)[0]
        RESULTS2[t] = brier_score(tr_y_structured, ts_y_structured_, S_list[:, eval_time], times=eval_time)[1][0]
        # # pdb.set_trace()
        # RESULTS1[t] = weighted_c_index(T_tr, E_tr, W_list[:, eval_time], T_ts, E_ts, eval_time)
        # RESULTS2[t] = weighted_brier_score(T_tr, E_tr, W_list[:, eval_time], T_ts, E_ts, eval_time)

    
    C_Index = np.round(RESULTS1, 4)
    Brier_Score = np.round(RESULTS2, 4)
    if not main_args.wandb_off:
        for i in range (len(C_Index)):
            wandb.log({f'Test: C_Index_{eval_times[i]}': C_Index[i]}, step=(epoch))
        for i in range (len(Brier_Score)):
            wandb.log({f'Test: Brier_Score_{eval_times[i]}': Brier_Score[i]}, step=(epoch))

        wandb.log({f'Test: C_Index_mean': C_Index.mean()}, step=(epoch))
        wandb.log({f'Test: Brier_Score_mean': Brier_Score.mean()}, step=(epoch))
    print(f'Test C_Index_mean: ',C_Index.mean())
    print(f'Test Brier_Score_mean: ',Brier_Score.mean())

    os.makedirs(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)
    f = open(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/internal_downstream_seed_{main_args.seed}.txt', 'a')
    f.write('epoch {}'.format(ee) + '\n')
    f.write(str(C_Index) + '\n')
    f.write('*' + str(C_Index.mean()) + '\n')
    
    f.write(str(Brier_Score) + '\n')
    f.write('*' + str(Brier_Score.mean()) + '\n\n')

    # result_list.append((C_Index.mean(), Brier_Score.mean()))

    # 예측값을 파일로 저장
    np.save(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/internal_test_predictions.npy', internal_test_predictions)




#############################External ONLY#############################
##test_step
eval_times = [3, 6, 12, 18, 24]
external_test_predictions = {}
model.eval()
with torch.no_grad():
    cost = 0.0
    T_tr=[]
    E_tr=[]
    T_ts=[]
    E_ts=[]
    
    S_list=[]
    W_list=[]
    for i, (t_mb, k_mb) in enumerate(train_eval_dataloader):
        T_tr.append(t_mb)
        E_tr.append(k_mb)
    T_tr=torch.cat(T_tr,dim=0)
    E_tr=torch.cat(E_tr,dim=0)
    
    T_tr = T_tr.cpu()
    E_tr = E_tr.cpu()
    for i, (ids, test_tabular, Test_data, t_mb, k_mb, m1_mb, m2_mb, m3_mb) in enumerate(external_dataloader):
    # for i, (Test_data, t_mb, label, _, _,proc_label) in enumerate(test_dataloader):  
        # X_ts = Test_data.to(device, dtype=torch.float)
        test_tabular = test_tabular.to(device, dtype=torch.float)
        t_mb = t_mb.to(device)
        k_mb = k_mb.to(device)
        m1_mb = m1_mb.to(device)
        m2_mb = m2_mb.to(device)
        m3_mb = m3_mb.to(device)
        T_ts.append(t_mb)
        E_ts.append(k_mb)

        # x = Test_data.detach().clone()

        out = model(test_tabular)
        log_surv_out = torch.log(1. - out + main_args.eps)
        H = log_surv_out.cumsum(1)
        S = torch.exp(H)
        W = 1-S
        # proc_label = proc_label.to(device)
        loss=nll_loss(out ,m1_mb ,m2_mb ,k_mb)
        # loss=nll_loss(out ,m1_mb ,m2_mb , k_mb) + rank(out ,m3_mb ,t_mb, k_mb) #rank에 sigma값 조정가능 default 0.5
        cost += loss

        # out = out.reshape(X_val.shape[0], args.num_category)
        # out = out.cpu()
        # H = out.cumsum(1)
        # S = torch.exp(-H)
        S_list.append(S)
        W_list.append(W)

        # 각 환자별로 데이터 저장   
        for idx in range(len(ids)):
            external_test_predictions[ids[idx]] = {
                'true_time': t_mb[idx].cpu(),
                'true_event': k_mb[idx].cpu(),
                'survival_probs': S[idx].cpu(),  # shape: (n_eval_times,)
                'hazard_probs': W[idx].cpu(),  # shape: (n_eval_times,)
                'eval_times': eval_times
            }
    T_ts=torch.cat(T_ts,dim=0)
    E_ts=torch.cat(E_ts,dim=0)
    
    S_list=torch.cat(S_list,dim=0)
    W_list=torch.cat(W_list,dim=0)

    T_ts = T_ts.cpu()
    E_ts = E_ts.cpu()
    S_list = S_list.cpu()
    W_list = W_list.cpu()



    cost = cost / len(test_dataloader)
    cost_value=cost.detach().cpu().numpy()
    # pdb.set_trace()
    if not main_args.wandb_off:
        wandb.log({f'Test Loss(External)': cost_value}, step=(epoch)) 
    print(f'Test(External): ',cost_value)

    tr_y_structured =  [(np.asarray(E_tr)[i], np.asarray(T_tr)[i]) for i in range(len(E_tr))]
    tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])


    ts_y_structured =  [(np.asarray(E_ts)[i], np.asarray(T_ts)[i]) for i in range(len(E_ts))]
    ts_y_structured = np.array(ts_y_structured, dtype=[('status', 'bool'),('time','<f8')])    

    # to compute brier-score without error
    T_ts_ = pd.DataFrame(T_ts, columns=['duration_'+main_args.spec_event])
    T_tr_ = pd.DataFrame(T_tr, columns=['duration_'+main_args.spec_event])
    # T_te_.loc[T_te_['duration_'+self.args.task] > T_tr_['duration_'+self.args.task].max(), 'event_time'] = T_tr_['duration_'+self.args.task].max()
    
    T_ts_.loc[T_ts_['duration_'+main_args.spec_event] > T_tr_['duration_'+main_args.spec_event].max()] = T_tr_['duration_'+main_args.spec_event].max()
    ts_y_structured_ =  [(np.asarray(E_ts)[i], np.asarray(T_ts_)[i]) for i in range(len(E_ts))]
    ts_y_structured_ = np.array(ts_y_structured_, dtype=[('status', 'bool'),('time','<f8')])

    RESULTS1 = np.zeros([len(eval_times)])
    RESULTS2 = np.zeros([len(eval_times)])
    
    for t, eval_time in enumerate(eval_times):
    
        RESULTS1[t] = concordance_index_ipcw(tr_y_structured, ts_y_structured, W_list[:, eval_time], tau=eval_time)[0]
        RESULTS2[t] = brier_score(tr_y_structured, ts_y_structured_, S_list[:, eval_time], times=eval_time)[1][0]
    
    
    C_Index_gbm = np.round(RESULTS1, 4)
    Brier_Score_gbm = np.round(RESULTS2, 4)
    if not main_args.wandb_off:
        for i in range (len(C_Index_gbm)):
            wandb.log({f'Test(External): C_Index_{eval_times[i]}': C_Index_gbm[i]}, step=(epoch))
        for i in range (len(Brier_Score_gbm)):
            wandb.log({f'Test(External): Brier_Score_{eval_times[i]}': Brier_Score_gbm[i]}, step=(epoch))

        wandb.log({f'Test(External): C_Index_mean': C_Index_gbm.mean()}, step=(epoch))
        wandb.log({f'Test(External): Brier_Score_mean': Brier_Score_gbm.mean()}, step=(epoch))
    print(f'Test C_Index_mean(External): ',C_Index_gbm.mean())
    print(f'Test Brier_Score_mean(External): ',Brier_Score_gbm.mean())

    os.makedirs(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)
    f = open(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/external_downstream_seed_{main_args.seed}.txt', 'a')
    f.write('epoch {}'.format(ee) + '\n')
    f.write(str(C_Index_gbm) + '\n')
    f.write('*' + str(C_Index_gbm.mean()) + '\n')
    
    f.write(str(Brier_Score_gbm) + '\n')
    f.write('*' + str(Brier_Score_gbm.mean()) + '\n\n')

    # 예측값을 파일로 저장
    np.save(f'./Test_result/tabular/{cohort}/{main_args.net_architect}_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/external_test_predictions.npy', external_test_predictions)




#     #############################GBM ONLY#############################
    
#     ##test_step
#     model.eval()
#     with torch.no_grad():
#         cost = 0.0
#         T_tr=[]
#         E_tr=[]
#         T_ts=[]
#         E_ts=[]
        
#         S_list=[]
#         W_list=[]
#         for i, (t_mb, k_mb) in enumerate(train_eval_dataloader):
#             T_tr.append(t_mb)
#             E_tr.append(k_mb)
#         T_tr=torch.cat(T_tr,dim=0)
#         E_tr=torch.cat(E_tr,dim=0)
        
#         T_tr = T_tr.cpu()
#         E_tr = E_tr.cpu()
#         for i, (Test_data, _, t_mb, k_mb, m1_mb, m2_mb, m3_mb) in enumerate(test_dataloader_gbm):
#             X_ts = Test_data.to(device, dtype=torch.float)
#             t_mb = t_mb.to(device)
#             k_mb = k_mb.to(device)
#             m1_mb = m1_mb.to(device)
#             m2_mb = m2_mb.to(device)
#             m3_mb = m3_mb.to(device)
#             T_ts.append(t_mb)
#             E_ts.append(k_mb)

#             out = model(X_ts)
#             H = out.cumsum(1)
#             S = torch.exp(-H)
#             W = 1-S
#             # proc_label = proc_label.to(device)
#             loss=nll_loss(out ,m1_mb ,m2_mb ,k_mb)
#             cost += loss

#             S_list.append(S)
#             W_list.append(W)
#         T_ts=torch.cat(T_ts,dim=0)
#         E_ts=torch.cat(E_ts,dim=0)
        
#         S_list=torch.cat(S_list,dim=0)
#         W_list=torch.cat(W_list,dim=0)

#         T_ts = T_ts.cpu()
#         E_ts = E_ts.cpu()
#         S_list = S_list.cpu()
#         W_list = W_list.cpu()



#         cost = cost / len(test_dataloader)
#         cost_value=cost.detach().cpu().numpy()

#         if not main_args.wandb_off:
#             wandb.log({f'Test Loss(Fold {fold_i})': cost_value}, step=(epoch+main_args.epochs*fold_i)) 
#         print(f'Test(Fold {fold_i}): ',cost_value)

#         tr_y_structured =  [(np.asarray(E_tr)[i], np.asarray(T_tr)[i]) for i in range(len(E_tr))]
#         tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])

#         ts_y_structured =  [(np.asarray(E_ts)[i], np.asarray(T_ts)[i]) for i in range(len(E_ts))]
#         ts_y_structured = np.array(ts_y_structured, dtype=[('status', 'bool'),('time','<f8')])    

#         # to compute brier-score without error
#         T_ts_ = pd.DataFrame(T_ts, columns=['duration_'+main_args.spec_event])
#         T_tr_ = pd.DataFrame(T_tr, columns=['duration_'+main_args.spec_event])

#         T_ts_.loc[T_ts_['duration_'+main_args.spec_event] > T_tr_['duration_'+main_args.spec_event].max()] = T_tr_['duration_'+main_args.spec_event].max()
#         ts_y_structured_ =  [(np.asarray(E_ts)[i], np.asarray(T_ts_)[i]) for i in range(len(E_ts))]
#         ts_y_structured_ = np.array(ts_y_structured_, dtype=[('status', 'bool'),('time','<f8')])

#         RESULTS1 = np.zeros([len(eval_times)])
#         RESULTS2 = np.zeros([len(eval_times)])
        
#         for t, eval_time in enumerate(eval_times):
        
#             RESULTS1[t] = concordance_index_ipcw(tr_y_structured, ts_y_structured, W_list[:, eval_time], tau=eval_time)[0]
#             RESULTS2[t] = brier_score(tr_y_structured, ts_y_structured_, S_list[:, eval_time], times=eval_time)[1][0]
        
        
#         C_Index_gbm = np.round(RESULTS1, 4)
#         Brier_Score_gbm = np.round(RESULTS2, 4)
#         if not main_args.wandb_off:
#             for i in range (len(C_Index_gbm)):
#                 wandb.log({f'Test(GBM)(Fold {fold_i}): C_Index_{eval_times[i]}': C_Index_gbm[i]}, step=(epoch+main_args.epochs*fold_i))
#             for i in range (len(Brier_Score_gbm)):
#                 wandb.log({f'Test(GBM)(Fold {fold_i}): Brier_Score_{eval_times[i]}': Brier_Score_gbm[i]}, step=(epoch+main_args.epochs*fold_i))

#             wandb.log({f'Test(GBM)(Fold {fold_i}): C_Index_mean': C_Index_gbm.mean()}, step=(epoch+main_args.epochs*fold_i))
#             wandb.log({f'Test(GBM)(Fold {fold_i}): Brier_Score_mean': Brier_Score_gbm.mean()}, step=(epoch+main_args.epochs*fold_i))
#         print(f'Test C_Index_mean(GBM)(Fold {fold_i}): ',C_Index_gbm.mean())
#         print(f'Test Brier_Score_mean(GBM)(Fold {fold_i}): ',Brier_Score_gbm.mean())

#         os.makedirs(f'./Test_result/tabular/{cohort}/',exist_ok=True)
#         f = open(f'./Test_result/tabular/{cohort}/' + main_args.net_architect + f"_GBM_{now.strftime('%Y-%m-%d_%H:%M:%S')}_downstream_Fold_{fold_i}.txt",'a')
#         f.write('epoch {}'.format(ee) + '\n')
#         f.write(str(C_Index_gbm) + '\n')
#         f.write('*' + str(C_Index_gbm.mean()) + '\n')
        
#         f.write(str(Brier_Score_gbm) + '\n')
#         f.write('*' + str(Brier_Score_gbm.mean()) + '\n\n')

#         result_list_gbm.append((C_Index_gbm.mean(), Brier_Score_gbm.mean()))


# # 전체 fold의 평균 성능 계산
# mean_C_Index = np.mean([result[0] for result in result_list])
# mean_Brier_Score = np.mean([result[1] for result in result_list])


# # 전체 fold 평균 성능을 로그에 기록
# if not main_args.wandb_off:
#     wandb.log({'Mean C_Index': mean_C_Index})
#     wandb.log({'Mean Brier_Score': mean_Brier_Score})

# print(f'Mean C_Index: {mean_C_Index}')
# print(f'Mean Brier_Score: {mean_Brier_Score}')


# # 전체 fold의 평균 성능 계산
# mean_C_Index_gbm = np.mean([result[0] for result in result_list_gbm])
# mean_Brier_Score_gbm = np.mean([result[1] for result in result_list_gbm])


# # 전체 fold 평균 성능을 로그에 기록
# if not main_args.wandb_off:
#     wandb.log({'Mean C_Index_gbm': mean_C_Index_gbm})
#     wandb.log({'Mean Brier_Score_gbm': mean_Brier_Score_gbm})

# print(f'Mean C_Index_gbm: {mean_C_Index_gbm}')
# print(f'Mean Brier_Score_gbm: {mean_Brier_Score_gbm}')