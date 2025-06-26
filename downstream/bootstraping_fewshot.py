import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import brier_score as brier_score_
from pycox.evaluation import EvalSurv
from utils_eval import weighted_c_index, weighted_brier_score, c_index, brier_score
import numba
from tqdm import tqdm
import pdb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)




def bootstraping(args, file_path, bootstrap_num, seed):
    eval_times = [3,6,12,24,48]

    pred = np.load(file_path, allow_pickle=True).item()
    if args.not_fewshot_npy:
        external_test_ids = np.load(f"{os.path.dirname(file_path)}/test_ids.npy", allow_pickle=True).tolist()
    else:
        external_test_ids = np.load("./code/load_trained_model/Test_result/tabular_dino/SNUH_severance_UCSF/external_few_shot/few_40/DINO_65_ex_seed_705_2025-01-23_22:21:18/test_ids.npy", allow_pickle=True).tolist()
        
    all_external_df = pd.read_csv(f'./data/UPenn_total_OS_all_v3_filtered.csv')
    external_df = all_external_df[all_external_df['id'].isin(external_test_ids)]

    pred = np.load(file_path, allow_pickle=True).item()
    eval_times = [3,6,12,18,24]
    bootstrap_external_results = []


    for i in range(bootstrap_num):
        bootstrap_external_data=external_df.sample(frac=1, replace=True, random_state=seed+i)

        E_ts = [pred[key]['true_event'] for key in bootstrap_external_data['id'].tolist() if key in pred]
        T_ts = [pred[key]['true_time'] for key in bootstrap_external_data['id'].tolist() if key in pred]
        S_list = [pred[key]['survival_probs'] for key in bootstrap_external_data['id'].tolist() if key in pred]
        W_list = [pred[key]['hazard_probs'] for key in bootstrap_external_data['id'].tolist() if key in pred]


        if isinstance(W_list[0], np.ndarray):
            W_list = [torch.tensor(w) if isinstance(w, np.ndarray) else w for w in W_list]
            S_list = [torch.tensor(s) if isinstance(s, np.ndarray) else s for s in S_list]

            S_list = torch.stack(S_list, dim=0)
            W_list = torch.stack(W_list, dim=0)

            T_ts_array = np.array(T_ts).flatten()
            E_ts_array = np.array(E_ts).flatten()

            S_array = np.array(S_list)
            W_array = np.array(W_list)

            
            time_specific_cindex_results = []
            brier_scores = []
            
            for k, eval_time in enumerate(eval_times):
                cindex = c_index(W_list[:, k].numpy(), T_ts_array, E_ts_array, eval_time)
                bscore = brier_score(W_list[:, k].numpy(), T_ts_array, E_ts_array, eval_time)
                
                time_specific_cindex_results.append((eval_time, cindex))
                brier_scores.append(bscore)
                
                print(f"Eval Time: {eval_time}, Concordance Index: {cindex}")
            print("Brier Scores:", brier_scores)
            row = {
                'bootstrap_iteration': i
            }
            
            for j, eval_time in enumerate(eval_times):
                row[f'{eval_time}_c_index'] = time_specific_cindex_results[j][1]
            
            # brier score 결과 추가
            for j, eval_time in enumerate(eval_times):
                row[f'{eval_time}_brier_score'] = brier_scores[j]
            
            bootstrap_external_results.append(row)



        else:
            S_list = torch.stack(S_list, dim=0)
            W_list = torch.stack(W_list, dim=0)

            T_ts_array = np.array(T_ts).flatten()
            E_ts_array = np.array(E_ts).flatten()

            S_array = np.array(S_list)
            W_array = np.array(W_list)

            time_specific_cindex_results = []
            brier_scores = []
            
            for k, eval_time in enumerate(eval_times):
                cindex = c_index(W_list[:, eval_time].numpy(), T_ts_array, E_ts_array, eval_time)
                bscore = brier_score(W_list[:, eval_time].numpy(), T_ts_array, E_ts_array, eval_time)
                
                time_specific_cindex_results.append((eval_time, cindex))
                brier_scores.append(bscore)
                
                print(f"Eval Time: {eval_time}, Concordance Index: {cindex}")
            print("Brier Scores:", brier_scores)
            row = {
                'bootstrap_iteration': i
            }
            
            for j, eval_time in enumerate(eval_times):
                row[f'{eval_time}_c_index'] = time_specific_cindex_results[j][1]
            
            for j, eval_time in enumerate(eval_times):
                row[f'{eval_time}_brier_score'] = brier_scores[j]
            
            bootstrap_external_results.append(row)

    external_results_table = pd.DataFrame(bootstrap_external_results)
    summary_row = {'bootstrap_iteration': 'mean'}
    std_row = {'bootstrap_iteration': 'std'}
    for eval_time in eval_times:
        # C-index
        summary_row[f'{eval_time}_c_index'] = external_results_table[f'{eval_time}_c_index'].mean()
        std_row[f'{eval_time}_c_index'] = external_results_table[f'{eval_time}_c_index'].std()
        
        # Brier score
        summary_row[f'{eval_time}_brier_score'] = external_results_table[f'{eval_time}_brier_score'].mean()
        std_row[f'{eval_time}_brier_score'] = external_results_table[f'{eval_time}_brier_score'].std()

    external_results_table = pd.concat([
        external_results_table, 
        pd.DataFrame([summary_row, std_row])
    ]).round(4)
    if args.not_fewshot_npy:
        os.makedirs(f"{os.path.dirname(file_path)}/bs_seed{args.seed}", exist_ok=True)
        external_results_table.to_csv(f"{os.path.dirname(file_path)}/bs_seed{args.seed}/external_few_bootstrap_results_table_{args.model_name}.csv", index=False)
    else:
        os.makedirs(f"{os.path.dirname(file_path)}/{args.external_seed}/bs_seed{args.seed}", exist_ok=True)
        external_results_table.to_csv(f"{os.path.dirname(file_path)}/{args.external_seed}/bs_seed{args.seed}/external_few_bootstrap_results_table_{args.model_name}.csv", index=False)


import argparse
import psutil
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Bootstraping')
    parser.add_argument('-c', '--cpu_start_num', default=0, type=int)
    parser.add_argument('--bootstrap_num', default=100, type=int)
    parser.add_argument('-s', '--seed', default=42, type=int)
    parser.add_argument('-f', '--file_path', type=str)
    parser.add_argument('-m', '--model_name', default='', type=str, help= 'only ML ex) _cox, _rsf' )
    parser.add_argument('--not_fewshot_npy', action='store_false')
    
    parser.add_argument('--external_seed', default=705, type=int)
    return parser

args = get_args_parser().parse_args()

p = psutil.Process()
p.cpu_affinity(range(args.cpu_start_num, args.cpu_start_num+4))

bootstraping(args, args.file_path, args.bootstrap_num, args.seed)
