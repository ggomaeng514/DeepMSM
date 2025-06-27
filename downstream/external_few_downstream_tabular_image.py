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
# from resnet_simclr import ResNetSimCLR
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
from utils_eval import weighted_c_index, weighted_brier_score, c_index, brier_score
from sklearn.preprocessing import StandardScaler

# for DINO
import argparse
import logging
import os
import sys
import numpy as np

from omegaconf import OmegaConf

import torch

import vision_transformer_3d as vits_3d
from load_VIT import *

import joblib

from datetime import datetime
now = datetime.now()


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
    
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--test_gpu_id', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-5) # ORIGINAL: 1e-4
    parser.add_argument('--seed', type=int, default=65) # 12347541
    parser.add_argument('--spec_patho', type=str, default='all') # 'GBL' # 
    parser.add_argument('--spec_duration', type=str, default='OS') # 'OS' # '1yr'
    parser.add_argument('--spec_event', type=str, default='death') # 'death' # 'prog'
    parser.add_argument('--dataset_name', type=str, default='SNUH_severance') # 'TCGA' # 
    parser.add_argument('--dataset_list', nargs='+', default=['SNUH', 'severance', 'UCSF'], help='selected_training_datasets') # ['UCSF','UPenn','TCGA','SNUH']]
    parser.add_argument('--remove_idh_mut', default=False, type=str2bool)
    parser.add_argument('--save_grad_cam', default=False, type=str2bool)
    parser.add_argument('--biopsy_exclusion', default=False, type=str2bool)
    parser.add_argument('--num_category', type=int, default=201)
    parser.add_argument('--net_architect', default='DINO', type=str) # 'SEResNext50'
    parser.add_argument('--cpu_start', '-c', type=int, default=20, help="bootstrapping number")
    parser.add_argument('--FP', action='store_true', help='use npy embedding')
    parser.add_argument('--finetuning', action='store_true', help='encoder finetuning')
    parser.add_argument('--cohort', type=str, choices=['all_data_embedding', 'korean_data_embedding', 'TCGA_data_embedding', 'UPennTCGA_data_embedding', 'UPenn_data_embedding'], default='UPennTCGA_data_embedding', help="cohort of operation: 'all_data_embedding' or 'korean_data_embedding' or 'TCGA_data_embedding' or 'UPennTCGA_data_embedding' or 'UPenn_data_embedding'")
    parser.add_argument('--wandb_off', action='store_true', help='wandb_off/on')
    
    parser.add_argument('--tabular_layer_depth', type=int, default=0, help="-1: None(nn.Identity())/ 0: linear / 1~: non-linear")
    parser.add_argument('--tabular_exdim', type=int, default=4, help="in tabular layer's expand dimension ratio (e.g. hidden_dim = 64 * tabular_exdim = 2 -> 128)")
    parser.add_argument('--tabular_layer_out_dim', type=int, default=64)

    parser.add_argument('--hazard_layer_depth', type=int, default=0, help="0: linear / 1~: non-linear")
    parser.add_argument('--hazard_exdim', type=int, default=4, help="in hazard layer's expand dimension ratio (e.g. hidden_dim = 64 * hazard_exdim = 2 -> 128)")
    parser.add_argument('--drop_rate', type=float, default=0.3, help="dropout rate: [0,1]")
    parser.add_argument('--gbm_only', action='store_true', help='grade 4 GBM only')
    parser.add_argument('--external_seed', type=int, default=631, help='external seed')
    parser.add_argument('--few_shot_num', type=int, default=40, help='external data number')
    parser.add_argument('--finetuning_layer', type=str, choices=['not_dino', 'only_hazard', 'last_linear'], default='not_dino', help='finetuning layer')
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


eval_times = [3, 6, 12, 18, 24]

p = psutil.Process()
p.cpu_affinity(range(main_args.cpu_start, main_args.cpu_start+5))

seed =main_args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
sklearn.random.seed(seed)


now = datetime.now()
if not main_args.wandb_off:
    WANDB_AUTH_KEY = 'your_wandb_auth_key_here'  # Replace with your actual WANDB_AUTH_KEY

    wandb.login(key=WANDB_AUTH_KEY)
    if main_args.gbm_only:
        wandb.init(project="mri_",
                name=f"tabular_image_DINO_external_few_downstream_{main_args.seed}_GBM_only",
                notes=f"{now.strftime('%Y-%m-%d_%H:%M:%S')}",
                tags = ["nonTCGA2korean","gbm_only", "external_few_shot"]
                )
    else:
        wandb.init(project="mri_",
                name=f"tabular_image_DINO_external_few_downstream_{main_args.seed}",
                notes=f"{now.strftime('%Y-%m-%d_%H:%M:%S')}",
                tags = ["nonTCGA2korean", "external_few_shot"]
                )

    print(f'args:{main_args.__dict__}')
    wandb.config.update(main_args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
PATH = './code/check/tabular_dino/external_few_shot/'
os.makedirs(PATH, exist_ok=True) 

combine_img(main_args)

main_args.device=device

cohort= "_".join(main_args.dataset_list)




X_external = pd.read_csv('./data/UPenn_total_OS_all_v3_filtered.csv')

X_train = X_external.sample(n=40, random_state=main_args.external_seed)

X_test = X_external.drop(X_train.index)

X_train_few = X_train.sample(n=main_args.few_shot_num, random_state=main_args.external_seed)

train_ids = X_train_few['id'].values
test_ids = X_test['id'].values  

os.makedirs(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)

# Save IDs
np.save(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/train_ids.npy', train_ids)
np.save(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/test_ids.npy', test_ids)




if main_args.FP:
    # scaler = StandardScaler()
    age_train_scaled = joblib.load(f'./age_scaler{main_args.seed}.pkl')

    train_dataset_external = Image_N_TabularDatasetFP(X_train_few, main_args, scaler = age_train_scaled)
    train_dataloader = DataLoader(train_dataset_external, batch_size = main_args.few_shot_num, shuffle=True, num_workers= 0, drop_last=False)

    test_dataset_external = Image_N_TabularDatasetFP(X_test, main_args, scaler = age_train_scaled)
    test_dataloader = DataLoader(test_dataset_external, batch_size = main_args.batch_size, shuffle=False, num_workers= 4, drop_last=False)

else:
    train_dataset_external = Image_N_TabularDataset(X_train_few, main_args, dataset_name=f'{main_args.dataset_name}', transforms=None, aug_transform=True, contrastive=False)
    train_dataloader = DataLoader(train_dataset_external, batch_size = main_args.few_shot_num, shuffle=True, num_workers= 0, drop_last=False)

    test_dataset_external = Image_N_TabularDataset(X_test, main_args, dataset_name=f'{main_args.dataset_name}', transforms=None, aug_transform=False, contrastive=False)
    test_dataloader = DataLoader(test_dataset_external, batch_size = main_args.batch_size, shuffle=False, num_workers= 4, drop_last=False)


if main_args.tabular_layer_depth == -1:
    tabular_layer = nn.Identity().to(device)
elif main_args.tabular_layer_depth == 0:
    tabular_layer = nn.Linear(len(main_args.data_key_list), main_args.tabular_layer_out_dim).to(device)
else:
    tabular_layer = MutilLayer_Outdim(input_dim= len(main_args.data_key_list), hidden_dim=main_args.tabular_layer_out_dim*main_args.tabular_exdim, out_dim=main_args.tabular_layer_out_dim , n_layers=main_args.tabular_layer_depth , drop_rate=main_args.drop_rate).to(device)




if main_args.FP:

    base_model = nn.Identity().to(device)
    model=Image_Tabular_HazardNet(main_args, base_model, tabular_layer, main_args.tabular_layer_out_dim).to(device)

    model_path = "./model/tabular_dino/SNUH_severance_UCSF/DINO_65_2025-02-25_17:10:40/tabular_DINO_2025-02-25_17:10:40_downstream_checkpoint.pth" # internal model path
    model.load_state_dict(torch.load(model_path))


'''Optimizer, Loss Function'''

if main_args.finetuning_layer == 'only_hazard':
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hazards_net.parameters():
        param.requires_grad = True
    optimizer = AdamP(model.hazards_net.parameters(), lr=main_args.lr, weight_decay=main_args.weight_decay)

    print("\n=== 파라미터 학습 상태 확인 ===")
    for name, param in model.named_parameters():
        print(f"{name}: requires_grad = {param.requires_grad}")
    print("============================\n")    

elif main_args.finetuning_layer == 'last_linear':
    
    for param in model.parameters():
        param.requires_grad = False
    for param in model.hazards_net.hazard_network[1].parameters():
        param.requires_grad = True
    optimizer = AdamP(model.hazards_net.hazard_network[1].parameters(), lr=main_args.lr, weight_decay=main_args.weight_decay)
else:
    optimizer = AdamP(model.parameters(), lr=main_args.lr, weight_decay=main_args.weight_decay)

scheduler = fetch_scheduler(optimizer, main_args)


class SetPartition:
    img_size_x = 120
    num_channels = 80

partition_cfg = SetPartition()

''' Training Function '''

start = time.time()
history = defaultdict(list)
model = model.to(device)

internal_test_predictions = {}
table_rows = []

threshold = 0
for epoch in tqdm(range(0,main_args.epochs)):
    print('Epoch {}/{}'.format(epoch, main_args.epochs))
    print('-' * 10)
    
    model.train()
    cost = 0.0
    
    for i, (ids,tabular_mb, x_mb, t_mb, k_mb, m1_mb, m2_mb, m3_mb) in enumerate(train_dataloader):
        
        tabular_mb = tabular_mb.to(device, dtype=torch.float)
        x_mb = x_mb.to(device, dtype=torch.float)
        t_mb = t_mb.to(device)
        k_mb = k_mb.to(device)
        m1_mb = m1_mb.to(device)
        m2_mb = m2_mb.to(device)
        m3_mb = m3_mb.to(device)

        out = model(tabular_mb, x_mb)

        loss=nll_loss(out ,m1_mb ,m2_mb ,k_mb)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss
    
    cost = cost / len(train_dataloader)
    cost_value=cost.detach().cpu().numpy()
    if not main_args.wandb_off:
        wandb.log({f'Train Loss': cost_value}, step=(epoch))
    print(f'Train_loss: ',cost_value)

    eval_times = [3, 6, 12, 18, 24]
    external_test_predictions = {}
    model.eval()
    with torch.no_grad():
        cost = 0.0
        T_ts=[]
        E_ts=[]
        
        S_list=[]
        W_list=[]
        external_test_results = []

        for i, (ids, test_tabular, Test_data, t_mb, k_mb, m1_mb, m2_mb, m3_mb) in enumerate(test_dataloader):
            X_ts = Test_data.to(device, dtype=torch.float)
            test_tabular = test_tabular.to(device, dtype=torch.float)
            t_mb = t_mb.to(device)
            k_mb = k_mb.to(device)
            m1_mb = m1_mb.to(device)
            m2_mb = m2_mb.to(device)
            m3_mb = m3_mb.to(device)
            T_ts.append(t_mb)
            E_ts.append(k_mb)

            out = model(test_tabular, X_ts)
            log_surv_out = torch.log(1. - out + main_args.eps)
            H = log_surv_out.cumsum(1)
            S = torch.exp(H)
            W = 1-S
            loss=nll_loss(out ,m1_mb ,m2_mb ,k_mb)
            cost += loss

            S_list.append(S)
            W_list.append(W)

            for idx in range(len(ids)):
                external_test_predictions[ids[idx]] = {
                    'true_time': t_mb[idx].cpu(),
                    'true_event': k_mb[idx].cpu(),
                    'survival_probs': S[idx].cpu(),
                    'hazard_probs': W[idx].cpu(),
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
            wandb.log({f'Test Loss(External)': cost_value}, step=(epoch)) 
        print(f'Test(External): ',cost_value)

        # 검토 필요
        T_ts_array = np.array(T_ts).flatten()
        E_ts_array = np.array(E_ts).flatten()

        time_specific_cindex_results = []
        brier_scores = []
        
        for k, eval_time in enumerate(eval_times):
            cindex = c_index(W_list[:, eval_time].numpy(), T_ts_array, E_ts_array, eval_time)
            bscore = brier_score(W_list[:, eval_time].numpy(), T_ts_array, E_ts_array, eval_time)
            
            time_specific_cindex_results.append((eval_time, cindex))
            brier_scores.append(bscore)
            
            print(f"Eval Time: {eval_time}, Concordance Index: {cindex}")
        print("Brier Scores:", brier_scores)
        
        row = {'epoch': epoch}        
        for j, eval_time in enumerate(eval_times):
            row[f'{eval_time}_c_index'] = time_specific_cindex_results[j][1]
        
        for j, eval_time in enumerate(eval_times):
            row[f'{eval_time}_brier_score'] = brier_scores[j]
        

        table_rows.append(row)
        external_results_table = pd.DataFrame(table_rows)

        mean_cindex = external_results_table.iloc[epoch,1:len(eval_times)+1].mean()
        mean_brier_score = external_results_table.iloc[epoch,len(eval_times)+1:].mean()
        
        if not main_args.wandb_off:
            for i in range (len(eval_times)):
                wandb.log({f'Test(External): C_Index_{eval_times[i]}': external_results_table.iloc[epoch, i+1]}, step=(epoch))
            for i in range (len(eval_times)):
                wandb.log({f'Test(External): Brier_Score_{eval_times[i]}': external_results_table.iloc[epoch, i+len(eval_times)+1]}, step=(epoch))        
            wandb.log({f'Test(External): C_Index_mean': mean_cindex}, step=(epoch))
            wandb.log({f'Test(External): Brier_Score_mean': mean_brier_score}, step=(epoch))
        
        os.makedirs(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)
        f = open(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/external_downstream_seed_{main_args.seed}.txt', 'a')
        f.write('epoch {}'.format(epoch) + '\n')
        f.write(str(external_results_table.iloc[epoch,1:len(eval_times)+1]) + '\n')
        f.write('*' + str(mean_cindex) + '\n')
        
        f.write(str(external_results_table.iloc[epoch,len(eval_times)+1:]) + '\n')
        f.write('*' + str(mean_brier_score) + '\n\n')

        np.save(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/external_test_predictions_epoch_{epoch}.npy', external_test_predictions)
        
        os.makedirs(f'./model/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)
        torch.save(model, f'./model/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/' + 'tabular_'+main_args.net_architect+f"_epoch_{epoch}_{now.strftime('%Y-%m-%d_%H:%M:%S')}_downstream_checkpoint.pth")
        if threshold < mean_cindex:
            print(threshold, mean_cindex)
            best_model = model.state_dict()

            threshold = mean_cindex
            print(epoch)
            f.write('^^^^^^^^^^^^^^^^^^best epoch {}^^^^^^^^^^^^^^^^^^'.format(epoch) + '\n')
            torch.save(model, f'./model/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/'+main_args.net_architect+f"_best_model_{now.strftime('%Y-%m-%d_%H:%M:%S')}_downstream_checkpoint.pth")
            np.save(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/external_test_predictions_best_model.npy', external_test_predictions)
            pd.DataFrame([external_results_table.iloc[epoch,:]], columns=external_results_table.columns).to_csv(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/' + main_args.net_architect+f"_results_best_epoch_{now.strftime('%Y-%m-%d_%H:%M:%S')}.csv", index=False)
external_results_table.to_csv(f'./Test_result/tabular_dino/{cohort}/external_few_shot/few_{main_args.few_shot_num}/{main_args.net_architect}_{main_args.seed}_ex_seed_{main_args.external_seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/' + 'tabular_'+main_args.net_architect+f"_results_table_{now.strftime('%Y-%m-%d_%H:%M:%S')}.csv", index=False)

