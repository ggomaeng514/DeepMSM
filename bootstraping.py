import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import brier_score as brier_score_
from pycox.evaluation import EvalSurv
from utils_eval import weighted_c_index, weighted_brier_score
import numba
from tqdm import tqdm
import pdb
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def auc_at_time(durations, events, survival_probs, eval_time):
    n_concordant = 0
    n_comparable = 0

    for i in range(len(durations)):
        if durations[i] <= eval_time and events[i] == 1:
            for j in range(len(durations)):
                if i != j:
                    if is_comparable(durations[i], durations[j], events[i], events[j]):
                        n_comparable += 1
                        if is_concordant(survival_probs[i], survival_probs[j], durations[i], durations[j], events[i], events[j]):
                            n_concordant += 1

    return n_concordant / n_comparable if n_comparable > 0 else np.nan

def time_dependent_cindex_with_details(durations, events, survival_probs, eval_times, S_all=True):
    weights = []
    aucs = []
    detailed_results = []
    for i, t in enumerate(eval_times):
        auc = auc_at_time(durations, events, survival_probs[:, t], t) if S_all else auc_at_time(durations, events, survival_probs[:, i], t)
        weight = np.sum((durations <= t) & (events == 1))
        aucs.append(auc)
        weights.append(weight)
        detailed_results.append(auc)

    weights = np.array(weights)
    aucs = np.array(aucs)
    cindex = np.sum(aucs * weights) / np.sum(weights) if np.sum(weights) > 0 else np.nan

    return cindex, detailed_results

@numba.jit(nopython=True)
def is_concordant(s_i, s_j, t_i, t_j, d_i, d_j):
    if t_i < t_j:
        return s_i < s_j
    return 0

@numba.jit(nopython=True)
def is_comparable(t_i, t_j, d_i, d_j):
    return (t_i < t_j and d_i == 1) or (t_i == t_j and d_i + d_j >= 1)
def brier_score_no_ipcw(S_df, T_ts_array, E_ts_array, eval_times):
    brier_scores = []
    for t in eval_times:
        delta_t = (T_ts_array <= t).astype(int)
        residuals = delta_t * (0 - S_df[t]) ** 2 + (1 - delta_t) * (1 - S_df[t]) ** 2
        
        brier_scores.append(residuals.mean())
    
    return brier_scores


### C(t)-INDEX CALCULATION
# https://github.com/chl8856/DeepHit/blob/master/utils_eval.py
def c_index_ps(Prediction, Time_survival, Death, Time):
    '''
        This is a cause-specific c(t)-index
        - Prediction      : risk at Time (higher --> more risky)
        - Time_survival   : survival/censoring time
        - Death           :
            > 1: death
            > 0: censored (including death from other cause)
        - Time            : time of evaluation (time-horizon when evaluating C-index)
    '''
    N = len(Prediction)
    A = np.zeros((N,N))
    Q = np.zeros((N,N))
    N_t = np.zeros((N,N))
    Num = 0
    Den = 0
    for i in range(N):
        A[i, np.where(Time_survival[i] < Time_survival)] = 1
        Q[i, np.where(Prediction[i] > Prediction)] = 1
  
        if (Time_survival[i]<=Time and Death[i]==1):
            N_t[i,:] = 1

    Num  = np.sum(((A)*N_t)*Q)
    Den  = np.sum((A)*N_t)

    if Num == 0 and Den == 0:
        result = -1 # not able to compute c-index!
    else:
        result = float(Num/Den)

    return result

### BRIER-SCORE
def brier_score_ps(Prediction, Time_survival, Death, Time):
    
    N = len(Prediction)
    y_true = ((Time_survival <= Time) * Death).astype(float)

    return np.mean((Prediction - y_true)**2)


def bootstraping(args,file_path, bootstrap_num, seed):
    eval_times = [3,6,12,24,48]

    pred = np.load(file_path, allow_pickle=True).item()

    test_ids = np.load(f"{os.path.dirname(file_path)}/test_ids.npy", allow_pickle=True).tolist()
    external_df = pd.read_csv(f'./data/UPenn_total_OS_all_v3_filtered.csv')
    test_ids_series = pd.Series(test_ids)

    tr_y_structured = pred['tr_y_structured']

    bootstrap_internal_results = []
    bootstrap_external_results = []

    for i in tqdm(range(bootstrap_num)):
        bootstrap_internal_data=test_ids_series.sample(frac=1, replace=True, random_state=seed+i)

        
        if 'T_tr_' in pred:
            T_tr_=pred['T_tr_']

        E_tr, T_tr = zip(*tr_y_structured)
        E_tr = np.asarray(E_tr)
        T_tr = np.asarray(T_tr)

        E_ts = [pred[key]['true_event'] for key in bootstrap_internal_data.tolist() if key in pred]
        T_ts = [pred[key]['true_time'] for key in bootstrap_internal_data.tolist() if key in pred]

        S_list = [pred[key]['survival_probs'] for key in bootstrap_internal_data.tolist() if key in pred]
        W_list = [pred[key]['hazard_probs'] for key in bootstrap_internal_data.tolist() if key in pred]
        
        if isinstance(W_list[0], np.ndarray):
            W_list = [torch.tensor(w) if isinstance(w, np.ndarray) else w for w in W_list]
            S_list = [torch.tensor(s) if isinstance(s, np.ndarray) else s for s in S_list]


            W_list = torch.stack(W_list, dim=0)
            S_list = torch.stack(S_list, dim=0)


            RESULTS1 = np.zeros([len(eval_times)])
            RESULTS2 = np.zeros([len(eval_times)])
            for t, eval_time in enumerate(eval_times):
                RESULTS1[t] = weighted_c_index(T_tr, E_tr, np.array(W_list[:, t]), torch.cat(T_ts).numpy(), torch.cat(E_ts).numpy(), eval_time)
                RESULTS2[t] = weighted_brier_score(T_tr, E_tr, np.array(W_list[:, t]), torch.cat(T_ts).numpy(), torch.cat(E_ts).numpy(), eval_time)


            bootstrap_internal_results.append({
                'bootstrap_iteration': i,
                'eval_times': eval_times,
                'concordance_index': RESULTS1.tolist(),
                'brier_score': RESULTS2.tolist()
            })  


        else:
            W_list = torch.stack(W_list, dim=0)
            S_list = torch.stack(S_list, dim=0)


            RESULTS1 = np.zeros([len(eval_times)])
            RESULTS2 = np.zeros([len(eval_times)])
            
            for t, eval_time in enumerate(eval_times):
                RESULTS1[t] = weighted_c_index(T_tr, E_tr, np.array(W_list[:, eval_time]), torch.cat(T_ts).numpy(), torch.cat(E_ts).numpy(), eval_time)
                RESULTS2[t] = weighted_brier_score(T_tr, E_tr, np.array(W_list[:, eval_time]), torch.cat(T_ts).numpy(), torch.cat(E_ts).numpy(), eval_time)
            bootstrap_internal_results.append({
                'bootstrap_iteration': i,
                'eval_times': eval_times,
                'concordance_index': RESULTS1.tolist(),
                'brier_score': RESULTS2.tolist()
            })  


    table_rows = []


    for result in bootstrap_internal_results:
        bootstrap_iteration = result['bootstrap_iteration']
        eval_times = result['eval_times']
        concordance_index = result['concordance_index']
        brier_score = result['brier_score']
        
        row = {'bootstrap_iteration': bootstrap_iteration}
        for i, eval_time in enumerate(eval_times):
            row[f'{eval_time}_c_index'] = concordance_index[i]
        
        for i, eval_time in enumerate(eval_times):
            row[f'{eval_time}_brier_score'] = brier_score[i]
        table_rows.append(row)

    results_table = pd.DataFrame(table_rows)

    # 결과 확인
    print(results_table)

    summary_row = {'bootstrap_iteration': 'mean'}
    std_row = {'bootstrap_iteration': 'std'}

    # 각 컬럼에 대해 평균과 표준편차 계산
    for eval_time in eval_times:
        # C-index
        summary_row[f'{eval_time}_c_index'] = results_table[f'{eval_time}_c_index'].mean()
        std_row[f'{eval_time}_c_index'] = results_table[f'{eval_time}_c_index'].std()
        
        # Brier score
        summary_row[f'{eval_time}_brier_score'] = results_table[f'{eval_time}_brier_score'].mean()
        std_row[f'{eval_time}_brier_score'] = results_table[f'{eval_time}_brier_score'].std()

    # 결과 테이블에 평균과 표준편차 행 추가
    results_table = pd.concat([
        results_table, 
        pd.DataFrame([summary_row, std_row])
    ]).round(4)
    results_table.to_csv(f"{os.path.dirname(file_path)}/internal_bootstrap_results_table{args.model_name}_ps_metric.csv", index=False)



    pred = np.load(file_path.replace(f'internal_test_predictions', f'external_test_predictions'), allow_pickle=True).item()
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
                cindex = c_index_ps(W_list[:, k].numpy(), T_ts_array, E_ts_array, eval_time)
                bscore = brier_score_ps(W_list[:, k].numpy(), T_ts_array, E_ts_array, eval_time)
                
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



        else:
            S_list = torch.stack(S_list, dim=0)
            W_list = torch.stack(W_list, dim=0)

            T_ts_array = np.array(T_ts).flatten()
            E_ts_array = np.array(E_ts).flatten()

            S_array = np.array(S_list)
            W_array = np.array(W_list)


            time_specific_cindex_results = []
            brier_scores = []
            
            # 각 평가 시점별로 c-index와 brier score 계산
            for k, eval_time in enumerate(eval_times):
                cindex = c_index_ps(W_list[:, eval_time].numpy(), T_ts_array, E_ts_array, eval_time)
                bscore = brier_score_ps(W_list[:, eval_time].numpy(), T_ts_array, E_ts_array, eval_time)
                
                time_specific_cindex_results.append((eval_time, cindex))
                brier_scores.append(bscore)
                
                print(f"Eval Time: {eval_time}, Concordance Index: {cindex}")
            print("Brier Scores:", brier_scores)
            row = {
                'bootstrap_iteration': i
            }
            
            # time-specific c-index 결과 추가
            for j, eval_time in enumerate(eval_times):
                row[f'{eval_time}_c_index'] = time_specific_cindex_results[j][1]
            
            # brier score 결과 추가
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
    external_results_table.to_csv(f"{os.path.dirname(file_path)}/external_bootstrap_results_table{args.model_name}_ps_metric.csv", index=False)

import argparse
import psutil
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Bootstraping')
    parser.add_argument('-c', '--cpu_start_num', default=0, type=int)
    parser.add_argument('--bootstrap_num', default=100, type=int)
    parser.add_argument('-s', '--seed', default=764, type=int)
    parser.add_argument('-f', '--file_path', type=str)
    parser.add_argument('-m', '--model_name', default='', type=str, help= 'only ML ex) _cox, _rsf' )
    return parser

args = get_args_parser().parse_args()

p = psutil.Process()
p.cpu_affinity(range(args.cpu_start_num, args.cpu_start_num+4))

bootstraping(args, args.file_path, args.bootstrap_num, args.seed)
