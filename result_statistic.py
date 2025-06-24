import pandas as pd
from scipy.stats import ttest_rel
from scipy.stats import sem, t
import numpy as np
import os
import argparse
import psutil
import pdb





def statistic_test(args, our, ml):
    if 'tabular_dino' in args.our_file_path:
        data_mode = 'tabular_image'
    elif 'tabular' in args.our_file_path:
        data_mode = 'tabular'    
    else:
        data_mode = 'image'
    if 'ML' in args.our_file_path:
        data_mode = 'ML'
    print(data_mode)

    if 'internal' in args.our_file_path:
        # 평균 컬럼 생성
        our["mean_c_index"] = our[["3_c_index", "6_c_index", "12_c_index", "24_c_index", "48_c_index"]].mean(axis=1)
        our["mean_brier_score"] = our[["3_brier_score", "6_brier_score", "12_brier_score", "24_brier_score", "48_brier_score"]].mean(axis=1)

        ml["mean_c_index"] = ml[["3_c_index", "6_c_index", "12_c_index", "24_c_index", "48_c_index"]].mean(axis=1)
        ml["mean_brier_score"] = ml[["3_brier_score", "6_brier_score", "12_brier_score", "24_brier_score", "48_brier_score"]].mean(axis=1)
        
    elif 'external' in args.our_file_path:
        # 평균 컬럼 생성
        our["mean_c_index"] = our[["3_c_index", "6_c_index", "12_c_index", "18_c_index", "24_c_index"]].mean(axis=1)
        our["mean_brier_score"] = our[["3_brier_score", "6_brier_score", "12_brier_score", "18_brier_score", "24_brier_score"]].mean(axis=1)

        ml["mean_c_index"] = ml[["3_c_index", "6_c_index", "12_c_index", "18_c_index", "24_c_index"]].mean(axis=1)
        ml["mean_brier_score"] = ml[["3_brier_score", "6_brier_score", "12_brier_score", "18_brier_score", "24_brier_score"]].mean(axis=1)

    # t-검정과 윌콕슨 검정 수행
    columns_to_test = our.columns
    test_results = []
    our=our.iloc[:-2]
    ml=ml.iloc[:-2]
    # pd.set_option('display.float_format', '{:.6f}'.format)
    for col in columns_to_test:
        our_col = our[col].values
        ml_col = ml[col].values
        
        # t-검정
        t_stat, t_p_val = ttest_rel(our_col, ml_col)
        
        # 각 집단의 평균과 표준 오차 계산
        our_mean = np.mean(our_col)
        ml_mean = np.mean(ml_col)
        
        our_std_err = sem(our_col)
        ml_std_err = sem(ml_col)
        
        # 자유도 계산
        df = len(our_col) - 1
        
        # t-분포의 임계값 (95% 신뢰구간)
        t_critical = t.ppf(0.975, df)
        
        # 각 집단의 신뢰구간 계산
        our_confidence_interval = (round(our_mean - t_critical * our_std_err, 4), round(our_mean + t_critical * our_std_err, 4))
        ml_confidence_interval = (round(ml_mean - t_critical * ml_std_err, 4), round(ml_mean + t_critical * ml_std_err, 4))
        
        # 결과 저장
        test_results.append({
            "Metric": col,
            "T-Stat": t_stat.round(4),
            "P-Value": t_p_val,
            "Our 95% CI": our_confidence_interval,
            "ML 95% CI": ml_confidence_interval
        })

    # 결과를 데이터프레임으로 정리
    test_results_df = pd.DataFrame(test_results)
    # import pdb; pdb.set_trace()
    # DL = args.our_file_path.split('/')[-2]
    
    # if 'internal' in args.our_file_path:
    #     test_results_df.to_csv(f'./Test_result/statistic_test/{seed}/{data_mode}/result_statistic_internal_{DL}_vs_{other_model_name}.csv', index=False)
    # elif 'external' in args.our_file_path:
    #     test_results_df.to_csv(f'./Test_result/statistic_test/{seed}/{data_mode}/result_statistic_external_{DL}_vs_{other_model_name}.csv', index=False)
    if 'few' in args.our_file_path:
        
        model_components = os.path.basename(args.other_file_path).split('_')
        other_model_name = '_'.join(model_components[4:]) 
        # pdb.set_trace()
        DL = os.path.basename(args.other_file_path).split('_')[-3] # 비교 대상 모델이 ML일때 image_tabular
        
        seed = args.our_file_path.split('/')[-2].split('_')[1]
        # few_num =  args.other_file_path.split('/')[-3]
        few_num =  args.other_file_path.split('/')[-4]
        if args.new_file_name:
            os.makedirs(os.path.dirname(args.new_file_name), exist_ok=True)
            test_results_df.to_csv(args.new_file_name, index=False)
        else:
            os.makedirs(f'./Test_result/statistic_test/{seed}/few_shot/{data_mode}', exist_ok=True)
            if 'internal' in args.our_file_path:
                test_results_df.to_csv(f'./Test_result/statistic_test/{seed}/few_shot/{data_mode}/result_statistic_internal_TI_40_vs_{other_model_name}_{few_num}_metric.csv', index=False)
            elif 'external' in args.our_file_path:
                test_results_df.to_csv(f'./Test_result/statistic_test/{seed}/few_shot/{data_mode}/result_statistic_external_TI_40_vs_{other_model_name}_{few_num}_metric.csv', index=False)

    else:
        # pdb.set_trace()
        # other_model_name = os.path.basename(args.other_file_path).split('_')[-1]
        other_model_name = os.path.basename(args.other_file_path).split('_')[-3]
        DL = os.path.basename(args.other_file_path).split('_')[-3] # 비교 대상 모델이 ML일때 image_tabular
        
        seed = args.our_file_path.split('/')[-2].split('_')[1]        
        if args.new_file_name:
            os.makedirs(os.path.dirname(args.new_file_name), exist_ok=True)
            test_results_df.to_csv(args.new_file_name, index=False)
        else:
            os.makedirs(f'./Test_result/statistic_test/{seed}/{data_mode}', exist_ok=True)
            if 'internal' in args.our_file_path:
                test_results_df.to_csv(f'./Test_result/statistic_test/{seed}/{data_mode}/result_statistic_internal_{DL}_vs_{other_model_name}_ps_metric.csv', index=False)
            elif 'external' in args.our_file_path:
                test_results_df.to_csv(f'./Test_result/statistic_test/{seed}/{data_mode}/result_statistic_external_{DL}_vs_{other_model_name}_ps_metric.csv', index=False)

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='Bootstraping')
    parser.add_argument('-c', '--cpu_start_num', default=0, type=int)
    parser.add_argument('-f', '--our_file_path', type=str)
    parser.add_argument('-o', '--other_file_path', type=str)
    
    parser.add_argument('--new_file_name', type=str, default=None)

    return parser

args = get_args_parser().parse_args()

p = psutil.Process()
p.cpu_affinity(range(args.cpu_start_num, args.cpu_start_num+4))
# import pdb; pdb.set_trace()
our=pd.read_csv(args.our_file_path, index_col=0)

ml=pd.read_csv(args.other_file_path, index_col=0)

statistic_test(args, our, ml)

