import numpy as np
import pandas as pd
import os 
import pickle
import joblib
import torch
import pdb
import psutil
import torch.nn as nn
import copy

import warnings
warnings.filterwarnings('ignore')

import json

from pathlib import Path

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, train_test_split

from plugin_models import CoxPH, RSF, CoxBoost, WeibullAFT
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

from load_VIT import *
from sksurv.metrics import concordance_index_ipcw, brier_score


if not os.path.exists('./results/'):
    os.makedirs('./results/')

if not os.path.exists('./storage/'):
    os.makedirs('./storage/')
    
filepath = './code/load_trained_model/ML_result'


# METHODS = ['cox', 'weibull', 'rsf', 'coxboost'] 
METHODS = ['cox',  'rsf', 'coxboost'] 
# METHODS = ['weibull'] 

import sys
sys.path.append("./code/load_trained_model")

from dataset import EditSurvDataset, EditProcSurvDataset, Eval_dataset,sample_minibatch_for_global_loss_opti_Subject_3D, Image_N_TabularDatasetFP, Image_N_TabularDataset
from utils import *
import monai
from monai.networks.nets import *

import sys
sys.path.append('./pretrain')
import resnet

import sklearn
import wandb
import psutil
import random
import argparse
from datetime import datetime
import copy

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Deep survival GBL: image only', add_help=add_help)
    
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--seed', type=int, default=65)
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
    parser.add_argument('--cpu_start', '-c', type=int, default=0, help="bootstrapping number")
    parser.add_argument('--FP', action='store_true', help='use npy embedding')
    parser.add_argument('--cohort', type=str, choices=['all_data_embedding', 'korean_data_embedding', 'TCGA_data_embedding', 'UPennTCGA_data_embedding', 'UPenn_data_embedding'], default='UPennTCGA_data_embedding', help="cohort of operation: 'all_data_embedding' or 'korean_data_embedding' or 'TCGA_data_embedding' or 'UPennTCGA_data_embedding' or 'UPenn_data_embedding'")
    parser.add_argument('--wandb_off', action='store_true', help='wandb_off/on')
    parser.add_argument('--data_type', type=str, choices=['tabular', 'image', 'tabular_image'], default='tabular', help="using data type")
    parser.add_argument('--image_pca_dim', type=int, default=10, help="pca output dimension / 0: not apply pca")
    parser.add_argument('--gbm_only', action='store_true', help='grade 4 GBM only')
    parser.add_argument('--training', default=True, type=str2bool)

    return parser


#%%
main_args = get_args_parser().parse_args()

main_args.compart_name = 'resized'
main_args.experiment_mode='none_partition'
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
p.cpu_affinity(range(main_args.cpu_start, main_args.cpu_start+3))

seed =main_args.seed
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
sklearn.random.seed(seed)

now = datetime.now()
if not main_args.wandb_off:
    WANDB_AUTH_KEY = 'your_wandb_auth_key_here'  # Replace with your actual WandB auth key

    wandb.login(key=WANDB_AUTH_KEY)
    if main_args.training:
        if main_args.gbm_only:
            wandb.init(project="mri_",
                    name=f"ML_{main_args.data_type}_downstream_{main_args.seed}_GBM_only",
                    notes=f"{now.strftime('%Y-%m-%d_%H:%M:%S')}",
                    tags = ["inference","gbm_only"]
                    )
        else:
            wandb.init(project="mri_",
                    name=f"ML_{main_args.data_type}_downstream_{main_args.seed}",
                    notes=f"{now.strftime('%Y-%m-%d_%H:%M:%S')}",
                    tags = ["inference"]
                    )
    else:
        if main_args.gbm_only:
            wandb.init(project="mri_",
                    name=f"IF_ML_{main_args.data_type}_downstream_{main_args.seed}_GBM_only",
                    notes=f"{now.strftime('%Y-%m-%d_%H:%M:%S')}",
                    tags = ["inference","gbm_only"]
                    )
        else:
            wandb.init(project="mri_",
                    name=f"IF_ML_{main_args.data_type}_downstream_{main_args.seed}",
                    notes=f"{now.strftime('%Y-%m-%d_%H:%M:%S')}",
                    tags = ["inference"]
                    )
    print(f'args:{main_args.__dict__}')
    wandb.config.update(main_args)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
PATH = './code/check/ML/'
os.makedirs(PATH, exist_ok=True) 

combine_img(main_args)

main_args.device=device

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from utils import K_fold 
from sklearn.decomposition import PCA


cohort= "_".join(main_args.dataset_list)
if main_args.gbm_only:
    df=pd.read_csv(f'./data/{cohort}_total_OS_GBM_v3_filtered.csv')
    
    train_valid_df, test_df = train_test_split(df, 
                                          test_size=0.2, 
                                          random_state=main_args.seed)

    train_df, valid_df = train_test_split(train_valid_df,
                                      test_size=0.25,
                                      random_state=main_args.seed)
else:
    df=pd.read_csv(f'./data/{cohort}_total_OS_all_v3_filtered.csv')
    train_valid_df, test_df = train_test_split(df, 
                                           test_size=0.2, 
                                           random_state=main_args.seed)

    train_df, valid_df = train_test_split(train_valid_df,
                                      test_size=0.25,
                                      random_state=main_args.seed)

train_ids = train_df['id'].values
valid_ids = valid_df['id'].values 
test_ids = test_df['id'].values

os.makedirs(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)

# Save IDs
np.save(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/train_ids.npy', train_ids)
np.save(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/valid_ids.npy', valid_ids)
np.save(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/test_ids.npy', test_ids)


internal_test_predictions={}
external_test_predictions={}

scaler = StandardScaler()

age_train_scaled = scaler.fit(train_df[['age']])

result = {
'C_Index' : [],
'Brier_Score' : []
}

if main_args.FP:
    train_dataset = Image_N_TabularDatasetFP(train_df, main_args, scaler = age_train_scaled)
    train_dataloader = DataLoader(train_dataset, batch_size = main_args.batch_size, shuffle=True, num_workers= 0, drop_last=False)

    train_eval_dataset = Eval_dataset(train_df, main_args, dataset_name=None)
    train_eval_dataloader = DataLoader(train_eval_dataset, batch_size = main_args.batch_size, shuffle=False, num_workers= 4, drop_last=False)
    
    valid_dataset = Image_N_TabularDatasetFP(valid_df, main_args, scaler = age_train_scaled)
    valid_dataloader = DataLoader(valid_dataset, batch_size = main_args.batch_size, shuffle=False, num_workers= 0, drop_last=False)

    test_dataset = Image_N_TabularDatasetFP(test_df, main_args, scaler = age_train_scaled)
    test_dataloader = DataLoader(test_dataset, batch_size = main_args.batch_size, shuffle=False, num_workers= 0, drop_last=False)


    external_df = pd.read_csv('./data/UPenn_total_OS_all_v3_filtered.csv')
    test_dataset_external = Image_N_TabularDatasetFP(external_df, main_args, scaler = age_train_scaled)
    test_dataloader_external = DataLoader(test_dataset_external, batch_size = main_args.batch_size, shuffle=False, num_workers= 4, drop_last=False)

else:
    train_dataset = Image_N_TabularDataset(train_df, main_args, main_args, dataset_name=main_args.dataset_name, transforms=None, aug_transform=False, contrastive=False)
    train_dataloader = DataLoader(train_dataset, batch_size = main_args.batch_size, shuffle=True, num_workers= 4, drop_last=False)

    train_eval_dataset = Eval_dataset(train_df, main_args, dataset_name=None)
    train_eval_dataloader = DataLoader(train_eval_dataset, batch_size = main_args.batch_size, shuffle=False, num_workers= 4, drop_last=False)

    valid_dataset = Image_N_TabularDataset(valid_df, main_args, dataset_name=f'{main_args.dataset_name}', transforms=None, aug_transform=False, contrastive=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size = main_args.batch_size, shuffle=False, num_workers= 4, drop_last=False)


    # test_dataset = EditProcSurvDataset(X_test, args, main_args, dataset_name=f'{main_args.dataset_name}',transforms=None, aug_transform=True, contrastive=False)
    test_dataset = Image_N_TabularDataset(test_df, main_args, main_args, dataset_name=main_args.dataset_name, transforms=None, aug_transform=False, contrastive=False)
    test_dataloader = DataLoader(test_dataset, batch_size = main_args.batch_size, shuffle=False, num_workers= 4, drop_last=False)


if main_args.FP:
    encoder = nn.Identity().to(device)

else:
    encoder_args = get_args_parser_in_notebook()
    cfg = OmegaConf.load(encoder_args.config_file)

    encoder, autocast_dtype = setup_and_build_model_3d(encoder_args, cfg)

if main_args.image_pca_dim != 0:
    pca = PCA(n_components=main_args.image_pca_dim, random_state=main_args.seed)

encoder.eval()
with torch.no_grad():
    tr_X=[]
    tr_f_X=[]
    vl_X=[]
    vl_f_X=[]
    te_X=[]
    te_f_X=[]
    te_X_ex=[]
    te_f_X_ex=[]

    T_tr=[]
    E_tr=[]
    T_vl=[]
    E_vl=[]
    T_ts=[]
    E_ts=[]
    T_ts_ex=[]
    E_ts_ex=[]

    in_ids_list=[]
    ex_ids_list=[]
    
    for i, (ids, x_tr_tabular, x_mb, t_mb, k_mb, _,_,_) in enumerate(train_dataloader):
        x_tr_tabular=x_tr_tabular.to(device, dtype=torch.float)
        x_mb = x_mb.to(device, dtype=torch.float)
        t_mb = t_mb.to(device)
        k_mb = k_mb.to(device)

        out = encoder(x_mb)

        tr_X.append(out)
        tr_f_X.append(x_tr_tabular)


        T_tr.append(t_mb)
        E_tr.append(k_mb)


    tr_X=torch.cat(tr_X,dim=0)
    tr_f_X=torch.cat(tr_f_X,dim=0)
    tr_T=torch.cat(T_tr,dim=0)
    tr_E=torch.cat(E_tr,dim=0)
    
        
    tr_X = tr_X.cpu()
    tr_f_X=tr_f_X.cpu()
    tr_T = tr_T.cpu()
    tr_E = tr_E.cpu()

    
    if main_args.data_type != 'tabular' and main_args.image_pca_dim != 0:
        tr_X = pca.fit_transform(tr_X) #pca

    if main_args.data_type == 'tabular' :
        print(f"{main_args.data_type} Mode")

    elif main_args.data_type == 'image' :
        print(f"{main_args.data_type} Mode")
        tr_f_X = tr_X

    elif main_args.data_type == 'tabular_image' :
        print(f"{main_args.data_type} Mode")
        tr_f_X = np.concatenate((tr_X,tr_f_X), axis=1)

    for i, (ids, x_vl_tabular,valid_data, t_mb, k_mb, _, _, _) in enumerate(valid_dataloader):
        x_vl_tabular=x_vl_tabular.to(device, dtype=torch.float)
        X_vl = valid_data.to(device, dtype=torch.float)
        t_mb = t_mb.to(device)
        k_mb = k_mb.to(device)

        out = encoder(X_vl)
        
        vl_X.append(out)
        vl_f_X.append(x_vl_tabular)

        T_vl.append(t_mb)
        E_vl.append(k_mb)

    vl_X=torch.cat(vl_X,dim=0)
    vl_f_X=torch.cat(vl_f_X,dim=0)
    vl_T=torch.cat(T_vl,dim=0)
    vl_E=torch.cat(E_vl,dim=0)

    vl_X = vl_X.cpu()
    vl_f_X = vl_f_X.cpu()
    vl_T = vl_T.cpu()
    vl_E = vl_E.cpu()

    if main_args.data_type != 'tabular' and main_args.image_pca_dim != 0:
        vl_X = pca.transform(vl_X) #pca


    if main_args.data_type == 'tabular' :
        print(f"{main_args.data_type} Mode")

    elif main_args.data_type == 'image' :
        print(f"{main_args.data_type} Mode")
        vl_f_X = vl_X

    elif main_args.data_type == 'tabular_image' :
        print(f"{main_args.data_type} Mode")
        vl_f_X = np.concatenate((vl_X,vl_f_X), axis=1)


    for i, (ids, x_te_tabular,Test_data, t_mb, k_mb, _, _, _) in enumerate(test_dataloader):
        x_te_tabular=x_te_tabular.to(device, dtype=torch.float)
        X_ts = Test_data.to(device, dtype=torch.float)
        t_mb = t_mb.to(device)
        k_mb = k_mb.to(device)

        out = encoder(X_ts)
        
        te_X.append(out)
        te_f_X.append(x_te_tabular)

        T_ts.append(t_mb)
        E_ts.append(k_mb)

        in_ids_list.extend(ids)
        for idx in range(len(ids)):
            internal_test_predictions[ids[idx]] = {
                'true_time': t_mb[idx].cpu(),
                'true_event': k_mb[idx].cpu(),
                'eval_times': eval_times
            }



    te_X=torch.cat(te_X,dim=0)
    te_f_X=torch.cat(te_f_X,dim=0)
    te_T=torch.cat(T_ts,dim=0)
    te_E=torch.cat(E_ts,dim=0)


    te_X = te_X.cpu()
    te_f_X = te_f_X.cpu()
    te_T = te_T.cpu()
    te_E = te_E.cpu()

    if main_args.data_type != 'tabular' and main_args.image_pca_dim != 0:
        te_X = pca.transform(te_X) #pca

    if main_args.data_type == 'tabular' :
        print(f"{main_args.data_type} Mode")

    elif main_args.data_type == 'image' :
        print(f"{main_args.data_type} Mode")
        te_f_X = te_X

    elif main_args.data_type == 'tabular_image' :
        print(f"{main_args.data_type} Mode")
        te_f_X = np.concatenate((te_X,te_f_X), axis=1)

    for i, (ids, x_te_tabular_ex,Test_data_ex, t_mb_gbm, k_mb_gbm, _, _, _) in enumerate(test_dataloader_external):
        x_te_tabular_ex=x_te_tabular_ex.to(device, dtype=torch.float)
        X_ts_ex = Test_data_ex.to(device, dtype=torch.float)
        t_mb_ex = t_mb_gbm.to(device)
        k_mb_ex = k_mb_gbm.to(device)

        out = encoder(X_ts_ex)
        
        te_X_ex.append(out)
        te_f_X_ex.append(x_te_tabular_ex)

        T_ts_ex.append(t_mb_ex)
        E_ts_ex.append(k_mb_ex)

        ex_ids_list.extend(ids)
        for idx in range(len(ids)):
            external_test_predictions[ids[idx]] = {
                'true_time': t_mb_ex[idx].cpu(),
                'true_event': k_mb_ex[idx].cpu(),
            }


    te_X_ex=torch.cat(te_X_ex,dim=0)
    te_f_X_ex=torch.cat(te_f_X_ex,dim=0)
    te_T_ex=torch.cat(T_ts_ex,dim=0)
    te_E_ex=torch.cat(E_ts_ex,dim=0)


    te_X_ex = te_X_ex.cpu()
    te_f_X_ex = te_f_X_ex.cpu()
    te_T_ex = te_T_ex.cpu()
    te_E_ex = te_E_ex.cpu()

    if main_args.data_type != 'tabular' and main_args.image_pca_dim != 0:
        te_X_ex = pca.transform(te_X_ex) #pca

    if main_args.data_type == 'tabular' :
        print(f"{main_args.data_type} Mode")

    elif main_args.data_type == 'image' :
        print(f"{main_args.data_type} Mode")
        te_f_X_ex = te_X_ex

    elif main_args.data_type == 'tabular_image' :
        print(f"{main_args.data_type} Mode")
        te_f_X_ex = np.concatenate((te_X_ex,te_f_X_ex), axis=1)

if main_args.training:
    set_alpha = [0.001, 0.01, 0.05, 0.1]
    set_n_estimators = [100, 200, 300, 400, 500]

    for method in METHODS:

        best_model = None
        best_cindex_avg = 0 
        best_params = {}

        print('{} training...'.format(method))
        
        tr_y_structured =  [(np.asarray(tr_E)[i], np.asarray(tr_T)[i]) for i in range(len(tr_E))]
        tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])

        vl_y_structured =  [(np.asarray(vl_E)[i], np.asarray(vl_T)[i]) for i in range(len(vl_E))]
        vl_y_structured = np.array(vl_y_structured, dtype=[('status', 'bool'),('time','<f8')])

        if method == 'cox':
            for alpha in set_alpha:
                tmp_model = CoxPH(alpha=alpha)
                tmp_model.fit(np.asarray(tr_f_X), pd.DataFrame(np.asarray(tr_T)), pd.DataFrame(np.asarray(tr_E)))
                tmp_pred = tmp_model.predict(np.asarray(vl_f_X), eval_times)

                cindexes = []
                for t, eval_time in enumerate(eval_times):
                    cindex = concordance_index_ipcw(tr_y_structured, vl_y_structured, tmp_pred[:, t], tau=eval_time)[0]
                    cindexes.append(cindex)
                
                cindex_avg = np.mean(cindexes)
                print(f'{method} | alpha: {alpha} | Avg C-index: {cindex_avg}')
                
                if cindex_avg > best_cindex_avg:
                    best_cindex_avg = cindex_avg
                    best_model = copy.deepcopy(tmp_model)
                    best_params = {'alpha': alpha}

        elif method == 'weibull':
            for alpha in set_alpha:
                tmp_model = WeibullAFT(alpha=alpha)
                tmp_model.fit(pd.DataFrame(np.asarray(tr_f_X)), pd.DataFrame(np.asarray(tr_T)), pd.DataFrame(np.asarray(tr_E)))
                tmp_pred = tmp_model.predict(pd.DataFrame(np.asarray(vl_f_X)), eval_times)

                cindexes = []
                for t, eval_time in enumerate(eval_times):
                    cindex = concordance_index_ipcw(tr_y_structured, vl_y_structured, tmp_pred[:, t], tau=eval_time)[0]
                    cindexes.append(cindex)

                cindex_avg = np.mean(cindexes)
                print(f'{method} | alpha: {alpha} | Avg C-index: {cindex_avg}')
                
                if cindex_avg > best_cindex_avg:
                    best_cindex_avg = cindex_avg
                    best_model = copy.deepcopy(tmp_model)
                    best_params = {'alpha': alpha}

        elif method == 'rsf':
            for n_estimators in set_n_estimators:
                tmp_model = RSF(n_estimators=n_estimators)
                tmp_model.fit(np.asarray(tr_f_X), pd.DataFrame(np.asarray(tr_T)), pd.DataFrame(np.asarray(tr_E)))
                tmp_pred = tmp_model.predict(np.asarray(vl_f_X), eval_times)

                cindexes = []
                for t, eval_time in enumerate(eval_times):
                    cindex = concordance_index_ipcw(tr_y_structured, vl_y_structured, tmp_pred[:, t], tau=eval_time)[0]
                    cindexes.append(cindex)

                cindex_avg = np.mean(cindexes)
                print(f'{method} | n_estimators: {n_estimators} | Avg C-index: {cindex_avg}')
                
                if cindex_avg > best_cindex_avg:
                    best_cindex_avg = cindex_avg
                    best_model = copy.deepcopy(tmp_model)
                    best_params = {'n_estimators': n_estimators}

        elif method == 'coxboost':
            for n_estimators in set_n_estimators:
                tmp_model = CoxBoost(n_estimators=n_estimators)
                tmp_model.fit(np.asarray(tr_f_X), pd.DataFrame(np.asarray(tr_T)), pd.DataFrame(np.asarray(tr_E)))
                tmp_pred = tmp_model.predict(np.asarray(vl_f_X), eval_times)

                cindexes = []
                for t, eval_time in enumerate(eval_times):
                    cindex = concordance_index_ipcw(tr_y_structured, vl_y_structured, tmp_pred[:, t], tau=eval_time)[0]
                    cindexes.append(cindex)

                cindex_avg = np.mean(cindexes)
                print(f'{method} | n_estimators: {n_estimators} | Avg C-index: {cindex_avg}')
                
                if cindex_avg > best_cindex_avg:
                    best_cindex_avg = cindex_avg
                    best_model = copy.deepcopy(tmp_model)
                    best_params = {'n_estimators': n_estimators}

        os.makedirs(filepath + '/storage/{}/{}/'.format(cohort, main_args.data_type), exist_ok=True)
        if main_args.gbm_only:
            joblib.dump(best_model, filepath + '/storage/{}/{}/{}2{}_{}_GBM_only.joblib'.format(cohort, main_args.data_type, main_args.cohort,cohort, method))
        else:
            joblib.dump(best_model, filepath + '/storage/{}/{}/{}2{}_{}.joblib'.format(cohort, main_args.data_type, main_args.cohort,cohort, method))

        print(f'Best {method} model saved with params: {best_params} and Avg C-index: {best_cindex_avg}')

RESULTS1 = np.zeros([len(METHODS), len(eval_times)]) 
RESULTS2 = np.zeros([len(METHODS), len(eval_times)]) 

tr_y_structured =  [(np.asarray(tr_E)[i], np.asarray(tr_T)[i]) for i in range(len(tr_E))]
tr_y_structured = np.array(tr_y_structured, dtype=[('status', 'bool'),('time','<f8')])

te_y_structured =  [(np.asarray(te_E)[i], np.asarray(te_T)[i]) for i in range(len(te_E))]
te_y_structured = np.array(te_y_structured, dtype=[('status', 'bool'),('time','<f8')])

# to compute brier-score without error
te_T_ = pd.DataFrame(te_T, columns=['duration_'+main_args.spec_event])
tr_T_ = pd.DataFrame(tr_T, columns=['duration_'+main_args.spec_event])

internal_test_predictions['tr_y_structured'] = tr_y_structured
internal_test_predictions['T_tr_'] = tr_T_

te_T_.loc[te_T_['duration_'+main_args.spec_event] > tr_T_['duration_'+main_args.spec_event].max()] = tr_T_['duration_'+main_args.spec_event].max()
te_y_structured_ =  [(np.asarray(te_E)[i], np.asarray(te_T_)[i]) for i in range(len(te_E))]
te_y_structured_ = np.array(te_y_structured_, dtype=[('status', 'bool'),('time','<f8')])



for m_idx, method in enumerate(METHODS):

    internal_test_predictions_copy = copy.deepcopy(internal_test_predictions)
    print('{} testing...'.format(method))

    if main_args.gbm_only:
        model = joblib.load(filepath + '/storage/{}/{}/{}2{}_{}_GBM_only.joblib'.format(cohort, main_args.data_type, main_args.cohort,cohort, method))
    else:
        model = joblib.load(filepath + '/storage/{}/{}/{}2{}_{}.joblib'.format(cohort, main_args.data_type, main_args.cohort,cohort, method))
    
    if method == 'weibull':
        pred  = model.predict(pd.DataFrame(np.asarray(te_f_X)), eval_times)
    else:
        pred  = model.predict(te_f_X, eval_times)
    
    for t, eval_time in enumerate(eval_times):
        RESULTS1[m_idx, t] = concordance_index_ipcw(tr_y_structured, te_y_structured, pred[:, t], tau=eval_time)[0]
        RESULTS2[m_idx, t] = brier_score(tr_y_structured, te_y_structured_, 1.- pred[:, t], times=eval_time)[1][0]

    for idx in range(len(in_ids_list)):
        internal_test_predictions_copy[in_ids_list[idx]]['hazard_probs'] = pred[idx]
        internal_test_predictions_copy[in_ids_list[idx]]['survival_probs'] = 1-pred[idx]

    os.makedirs(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)
    np.save(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/internal_test_predictions_{method}.npy', internal_test_predictions_copy)


main_c_index_result_df=pd.DataFrame(RESULTS1, index=METHODS, columns=eval_times)
main_brier_result_df=pd.DataFrame(RESULTS2, index=METHODS, columns=eval_times)


main_c_index_result_df.to_csv(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/c_index_internal_result.csv')
main_brier_result_df.to_csv(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/brier_internal_result.csv')

mean_RESULTS1 = np.mean(RESULTS1, axis=1)
mean_RESULTS2 = np.mean(RESULTS2, axis=1)
Concordance_Index = np.round(mean_RESULTS1, 4)
Brier_Score = np.round(mean_RESULTS2, 4)

if not main_args.wandb_off:
    for m_idx, method in enumerate(METHODS):
        state = {
                f'Test:C_index_{method}': main_c_index_result_df.iloc[m_idx].tolist(),
                f'Test:Brier_score_{method}' : main_brier_result_df.iloc[m_idx].tolist(),
                f'Test:Concordance_Index_Mean_{method}' : main_c_index_result_df.iloc[m_idx].mean(),
                f'Test:Brier_Score_Mean_{method}' : main_brier_result_df.iloc[m_idx].mean(),
                }

        wandb.log(state)

#############################EXTERNAL#############################
eval_times = [3, 6, 12, 18, 24]

te_y_structured_ex =  [(np.asarray(te_E_ex)[i], np.asarray(te_T_ex)[i]) for i in range(len(te_E_ex))]
te_y_structured_ex = np.array(te_y_structured_ex, dtype=[('status', 'bool'),('time','<f8')])

te_T_ex_ = pd.DataFrame(te_T_ex, columns=['duration_'+main_args.spec_event])
tr_T_ = pd.DataFrame(tr_T, columns=['duration_'+main_args.spec_event])

te_T_ex_.loc[te_T_ex_['duration_'+main_args.spec_event] > tr_T_['duration_'+main_args.spec_event].max()] = tr_T_['duration_'+main_args.spec_event].max()
te_y_structured_ex_ =  [(np.asarray(te_E_ex)[i], np.asarray(te_T_ex_)[i]) for i in range(len(te_E_ex))]
te_y_structured_ex_ = np.array(te_y_structured_ex_, dtype=[('status', 'bool'),('time','<f8')])


for m_idx, method in enumerate(METHODS):

    external_test_predictions_copy = copy.deepcopy(external_test_predictions)
    print('{} testing...'.format(method))

    if main_args.gbm_only:
        model = joblib.load(filepath + '/storage/{}/{}_{}2{}_{}_GBM_only.joblib'.format(main_args.dataset_name, main_args.data_type, main_args.cohort,cohort, method))
    else:
        model = joblib.load(filepath + '/storage/{}/{}_{}2{}_{}.joblib'.format(main_args.dataset_name, main_args.data_type, main_args.cohort,cohort, method))
    
    if method == 'weibull':
        pred  = model.predict(pd.DataFrame(np.asarray(te_f_X_ex)), eval_times)
    else:
        pred  = model.predict(te_f_X_ex, eval_times)
    
    for t, eval_time in enumerate(eval_times):
        RESULTS1[m_idx, t] = concordance_index_ipcw(tr_y_structured, te_y_structured_ex, pred[:, t], tau=eval_time)[0]
        RESULTS2[m_idx, t] = brier_score(tr_y_structured, te_y_structured_ex_, 1.- pred[:, t], times=eval_time)[1][0]

    for idx in range(len(ex_ids_list)):
        external_test_predictions_copy[ex_ids_list[idx]]['hazard_probs'] = pred[idx]
        external_test_predictions_copy[ex_ids_list[idx]]['survival_probs'] = 1-pred[idx]
        external_test_predictions_copy[ex_ids_list[idx]]['eval_times'] = eval_times

    os.makedirs(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}',exist_ok=True)
    np.save(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/external_test_predictions_{method}.npy', external_test_predictions_copy)

main_c_index_result_df_ex=pd.DataFrame(RESULTS1, index=METHODS, columns=eval_times)
main_brier_result_df_ex=pd.DataFrame(RESULTS2, index=METHODS, columns=eval_times)


main_c_index_result_df_ex.to_csv(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/c_index_external_result.csv')
main_brier_result_df_ex.to_csv(f'./Test_result/ML_{main_args.data_type}/{cohort}/ML_{main_args.seed}_{now.strftime("%Y-%m-%d_%H:%M:%S")}/brier_external_result.csv')

mean_RESULTS1_ex = np.mean(RESULTS1, axis=1)
mean_RESULTS2_ex = np.mean(RESULTS2, axis=1)
Concordance_Index_ex = np.round(mean_RESULTS1_ex, 4)
Brier_Score_ex = np.round(mean_RESULTS2_ex, 4)

if not main_args.wandb_off:
    for m_idx, method in enumerate(METHODS):
        state = {
                f'Test(External):C_index_{method}': main_c_index_result_df_ex.iloc[m_idx].tolist(),
                f'Test(External):Brier_score_{method}' : main_brier_result_df_ex.iloc[m_idx].tolist(),
                f'Test(External):Concordance_Index_Mean_{method}' : main_c_index_result_df_ex.iloc[m_idx].mean(),
                f'Test(External):Brier_Score_Mean_{method}' : main_brier_result_df_ex.iloc[m_idx].mean(),
                }

        wandb.log(state)
