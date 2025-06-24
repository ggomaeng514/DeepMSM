import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch.optim as optim
from torch.optim import lr_scheduler
from torchsummary import summary as summary
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import torchio as tio

import os
import glob
import copy
from datetime import datetime
import cv2
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import json
import argparse
import nibabel as nib


from monai.transforms import RandFlip, Rand3DElastic, RandAffine, RandGaussianNoise, AdjustContrast, RandSpatialCrop # Rand3DElastic
from sklearn.model_selection import train_test_split, StratifiedKFold
from distutils.dir_util import copy_tree
from tqdm import tqdm
from PIL import Image

import scipy.interpolate
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import matplotlib.animation as animation
import plotly.graph_objects as go
from IPython.display import HTML

from lifelines.utils import concordance_index
from sklearn.utils import resample
from sksurv.metrics import integrated_brier_score
from sksurv.metrics import brier_score
from skimage.transform import resize

from dataset import sample_minibatch_for_global_loss_opti_Subject_3D

import sys
sys.path.append('./pretrain')
import pdb
#%%

''' lambda '''
to_np = lambda x: x.detach().cpu().numpy()
to_cuda = lambda x: torch.from_numpy(x).float().device()

# get_dataset_name = lambda dataset_list: '_'.join(dataset_list)
get_dir = lambda directory: [dir for dir in os.listdir(directory) if os.path.isdir(os.path.join(directory, dir))]
convert_list2df = lambda idlist: pd.DataFrame(idlist, columns = ['ID'], dtype='string').set_index('ID')

#%%

def print_args(args, exp_path):
  args_path = os.path.join(exp_path,'commandline_args.txt')
  with open(args_path, 'w') as f:
      json.dump(args.__dict__, f, indent=2)

  with open(args_path, 'r') as f:
      args.__dict__ = json.load(f)

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
# set_seed(args.seed)

def min_max_norm(img):
  
  img = img.astype(np.float32)
  img = (img-img.min())/(img.max()-img.min())
  img = (img*255).astype(np.uint8)
  img = np.stack((img,)*3, axis=-1)
  
  return img

def superimpose_img(img, heatmap, alpha = 0.3):
  
  grad_heatmap = resize(heatmap, (img.shape[0], img.shape[1], img.shape[2]))

  cmap = plt.cm.jet
  grad_heatmap_rgb = cmap(grad_heatmap)
  grad_heatmap_rgb = grad_heatmap_rgb[...,:3]
  grad_heatmap_rgb = np.uint8(grad_heatmap_rgb * 255)

  grad_result = grad_heatmap_rgb * alpha + img * (1 - alpha) #.astype(np.uint8)
  grad_result = grad_result / np.max(grad_result)

  return grad_result, grad_heatmap

def plot_slices_superimposed(data, x_slice, y_slice, z_slice, use_midline = True):
    
    matplotlib.rcParams['animation.embed_limit'] = 500
    
    # get the x, y, z coordinates
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    print(f'x:{x}, y:{y}, z:{z}')

    if use_midline:
      xslice = data.shape[0] // 2 # specify
      yslice = data.shape[1] // 2 # specify
      zslice = data.shape[2] // 2 # specify
    else:
      xslice = x_slice
      yslice = y_slice
      zslice = z_slice

    print(f'xslice:{xslice}, yslice:{yslice}, zslice:{zslice}')

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define meshgrid
    x, y, z = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    
    # Take slices
    mask_x = np.abs(x - xslice) < 0.5
    mask_y = np.abs(y - yslice) < 0.5
    mask_z = np.abs(z - zslice) < 0.5
    mask = mask_x | mask_y | mask_z

    # Plot slices with alpha = 0.5 for some transparency
    scatter = ax.scatter(x[mask], y[mask], z[mask], c=data[mask], s=20, cmap = 'gray') # norm=norm, # , cmap = 'jet'

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Invert axis
    # ax.invert_xaxis()
    ax.invert_yaxis()
    # ax.invert_zaxis()
    
    # make 2 rotations with different directions
    total_frames = 720  # for two rounds # 360 #

    def update(num):
        
        # Angles for first rotation
        final_azim_1 = 300
        final_elev_1 = 285

        # Angles for second rotation
        final_azim_2 = 600
        final_elev_2 = 570

        if num < total_frames / 2:
            azim = (final_azim_1 / (total_frames / 2)) * num
            elev = (final_elev_1 / (total_frames / 2)) * num
        else:
            azim = final_azim_1 + ((final_azim_2 - final_azim_1) / (total_frames / 2)) * (num - total_frames / 2)
            elev = final_elev_1 + ((final_elev_2 - final_elev_1) / (total_frames / 2)) * (num - total_frames / 2)

        ax.view_init(elev=elev, azim=azim)
        return scatter
    
   
    ani = animation.FuncAnimation(fig, update, frames=np.arange(0, total_frames, 1), interval=60)
    html = ani.to_jshtml() # save as html 
    
    # return ani # save as gif
    return html#, ani

def plotly_slices_superimposed(data):
    
    # get the x, y, z coordinates
    x = np.arange(data.shape[0])
    y = np.arange(data.shape[1])
    z = np.arange(data.shape[2])
    print(f'x:{x}, y:{y}, z:{z}')

    # Slice indices
    xslice = data.shape[0] // 2 # specify
    yslice = data.shape[1] // 2 # specify
    zslice = data.shape[2] // 2 # specify
    print(f'xslice:{xslice}, yslice:{yslice}, zslice:{zslice}')

    # Define meshgrid
    x, y, z = np.mgrid[:data.shape[0], :data.shape[1], :data.shape[2]]
    
    # Take slices
    mask_x = np.abs(x - xslice) < 0.5
    mask_y = np.abs(y - yslice) < 0.5
    mask_z = np.abs(z - zslice) < 0.5
    mask = mask_x | mask_y | mask_z

    # Ensure there's data to plot
    
    fig = go.Figure(data=[go.Scatter3d(
        x=x[mask].flatten(),
        y=y[mask].flatten(),
        z=z[mask].flatten(),
        mode='markers',
        marker=dict(
            size=5,
            opacity=0.8,
            color=data[mask].flatten()
        )
    )])

    # tight layout
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
    fig.show()

#%%

def get_n_intervals(fixed_interval_width = False):

  if fixed_interval_width:
    breaks=np.arange(0.,365.*5,365./8)
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    # print(f'n_intervals: {n_intervals}') # 19
  else:
    halflife=365.*2
    breaks=-np.log(1-np.arange(0.0,0.96,0.05))*halflife/np.log(2) 
    n_intervals=len(breaks)-1
    timegap = breaks[1:] - breaks[:-1]
    # print(f'n_intervals: {n_intervals}') # 19

    return breaks, n_intervals

# get 95% confidence interval of concordance index using bootstrap
def bootstrap_cindex(time, prediction, event, n_iterations=1000):
    # Compute the original C-index
    original_c_index = concordance_index(time, prediction, event)

    # Initialize a list to store bootstrapped C-indexes
    bootstrap_c_indexes = []

    # Perform bootstrapping
    for i in range(n_iterations):
        # Resample with replacement
        resample_indices = resample(np.arange(len(time)), replace=True)
        time_sample = time[resample_indices]
        event_sample = event[resample_indices]
        prediction_sample = prediction[resample_indices]

        # Compute the C-index on the bootstrap sample
        c_index_sample = concordance_index(time_sample, prediction_sample, event_sample)

        bootstrap_c_indexes.append(c_index_sample)

    # Compute the 95% confidence interval for the C-index
    ci_lower = np.percentile(bootstrap_c_indexes, 2.5)
    ci_upper = np.percentile(bootstrap_c_indexes, 97.5)

    return original_c_index, ci_lower, ci_upper

#%%

''' copying and combining images of dataset_list '''

def combine_img(main_args):
  dataset_name = '_'.join(main_args.dataset_list)
  
  if main_args.net_architect =='VisionTransformer':
    target_dataset_path = os.path.join(main_args.data_dir, dataset_name, "VIT",f'{main_args.compart_name}_BraTS_v2')
    os.makedirs(target_dataset_path, exist_ok=True)
  
  elif main_args.net_architect == 'DenseNet' or 'resnet50_cbam' or 'SEResNext50' or 'pretrained_resnet':
    target_dataset_path = os.path.join(main_args.data_dir, dataset_name, f'{main_args.compart_name}_BraTS_v2')
    os.makedirs(target_dataset_path, exist_ok=True)
  
  if len(os.listdir(target_dataset_path)) != 0:
    print(f"Already copyied images of {dataset_name} for training to {target_dataset_path} path")
  
  else:
    for dataset in main_args.dataset_list:
      print(f"copying images of {dataset} for training to {target_dataset_path} path")
      
      if main_args.net_architect == 'VisionTransformer':
        img_dataset_path = os.path.join(main_args.data_dir, dataset, "VIT",f'{main_args.compart_name}_BraTS_v2')
      elif main_args.net_architect == 'DenseNet' or 'resnet50_cbam' or 'SEResNext50' or 'pretrained_resnet':
        img_dataset_path = os.path.join(main_args.data_dir, dataset, f'{main_args.compart_name}_BraTS_v2')
      
      for img_dir in tqdm(os.listdir(img_dataset_path)):
        img_dir_path = os.path.join(img_dataset_path, img_dir)
        print(f'img_dir_path:{img_dir_path}')
        os.makedirs(os.path.join(target_dataset_path, img_dir), exist_ok=True)
        copy_tree(img_dir_path, os.path.join(target_dataset_path, img_dir))

#%%

''' getting together multiple (i.e. SNUH, severance, UPenn) ${dataset}_OS_all.csv files into final csv indexing only 1) GBL vs all; and 2) 1yr vs OS, and save them into anoter .csv file '''

def save_label_dataset_list(main_args, args):
  
  df = pd.DataFrame()
  
  for dataset in main_args.dataset_list:
    print(f'dataset:{dataset} for training')
    
    if args.net_architect == 'VisionTransformer':
      df_dataset_path = os.path.join(args.label_dir, f'{dataset}_OS_all_vit.csv')
    
    elif args.net_architect == 'DenseNet' or 'resnet50_cbam' or 'SEResNext50':
      df_dataset_path = os.path.join(args.label_dir, f'{dataset}_OS_all.csv')
    
    df_data = pd.read_csv(df_dataset_path, dtype='string') # , index_col=0, dtype='string') # int: not working
    df_data = df_data.set_index('ID')
    df_data = df_data.sort_index(ascending=True)
        
    df_dataset = df_data[args.data_key_list]
    print(f'df_dataset.shape:{df_dataset.shape}')
    # df_label_dataset_list = pd.merge(df_dataset, df_label_dataset_list, left_on='ID', right_index=True) # NOT WORKING
    df = pd.concat([df_dataset, df])
  print(f'df_label_dataset_list.shape:{df.shape}') # 
  # print(f'df.head:{df.head(10)}')

  dataset_name = '_'.join(main_args.dataset_list)
  
  # ref: https://wooono.tistory.com/293

  if main_args.spec_patho == 'GBL':
    print(f'filtering before GBL; {len(df.index.values)} cases')
    condition = df.GBL.astype(int) == 1 # 1 means mut, not 0 
    filtered_ID = df[condition].index.tolist() 
    df = df.loc[sorted(filtered_ID),:]
    print(f'filtering after GBL; {len(df.index.values)} cases')

  if main_args.biopsy_exclusion:
    print(f'filtering before biopsy_exclusion; {len(df.index.values)} cases')
    print(f'df.columns:{df.columns}')
    if "biopsy_exclusion" in df.columns:
      condition = df.biopsy_exclusion.astype(int) == 0 # 1 means biopsy exclusion, not 0 
      filtered_ID = df[condition].index.tolist() 
      df = df.loc[sorted(filtered_ID),:]
      print(f'filtering after biopsy_exclusion; {len(df.index.values)} cases')

  if main_args.spec_event == 'death':
    if main_args.spec_duration == '1yr':
        df = df.astype({'event_death': 'int'})
        print('events before 1yr:')
        print(df['event_death'].sum())
        df.loc[(df['event_death'] == 1) & (df['duration_death'].astype(int) > 365), 'event_death'] = 0
        print(f'events after 1yr:')
        print(df['event_death'].sum())
    
    else:
        pass

  elif main_args.spec_event == 'prog':

    if main_args.spec_duration == '1yr':
        df = df.astype({'event_prog': 'int'})
        print('events before 1yr:')
        print(df['event_prog'].sum())
        df.loc[(df['event_prog'] == 1) & (df['duration_prog'].astype(int) > 365), 'event_prog'] = 0
        print(f'events after 1yr:')
        print(df['event_prog'].sum())
        
    else:
        pass

  df_path = os.path.join(args.label_dir, f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
  
  if args.net_architect =='VisionTransformer':
    df_path = os.path.join(args.label_dir, f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_vit.csv')
  
  df.to_csv(df_path)
  
  print(f'saving new label csv file for {dataset_name} at {df_path}') # 

  return df

def save_label_ext_dataset(main_args, args):
  
  ext_df = pd.DataFrame()
  
  print(f'dataset:{main_args.ext_dataset_name} for training')
  
  if args.net_architect =='VisionTransformer':
    ext_df_dataset_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_OS_all_vit.csv')
  elif args.net_architect =='DenseNet' or 'resnet50_cbam' or 'SEResNext50':
    ext_df_dataset_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_OS_all.csv')
  
  ext_df_data = pd.read_csv(ext_df_dataset_path, dtype='string') # , index_col=0, dtype='string') # int: not working
  ext_df_data = ext_df_data.set_index('ID')
  ext_df_data = ext_df_data.sort_index(ascending=True)
    
  ext_df = ext_df_data[args.data_key_list]
  print(f'ext_df_dataset.shape:{ext_df.shape}')
  
  # ref: https://wooono.tistory.com/293

  if main_args.spec_patho == 'GBL':
    print(f'filtering before GBL; {len(ext_df.index.values)} cases')
    condition = ext_df.GBL.astype(int) == 1
    filtered_ID = ext_df[condition].index.tolist() 
    ext_df = ext_df.loc[sorted(filtered_ID),:]
    print(f'filtering after GBL; {len(ext_df.index.values)} cases')

  if main_args.biopsy_exclusion:
    print(f'filtering before biopsy_exclusion; {len(ext_df.index.values)} cases')
    if "biopsy_exclusion" in ext_df.columns:
      condition = ext_df.biopsy_exclusion.astype(int) == 0 # 1 means biopsy exclusion, not 0 
      filtered_ID = ext_df[condition].index.tolist() 
      ext_df = ext_df.loc[sorted(filtered_ID),:]
      print(f'filtering after biopsy_exclusion; {len(ext_df.index.values)} cases')

  if main_args.spec_event == 'death':
    if main_args.spec_duration == '1yr':
        ext_df = ext_df.astype({'event_death': 'int'})
        print('events before 1yr:')
        print(ext_df['event_death'].sum())
        ext_df.loc[(ext_df['event_death'] == 1) & (ext_df['duration_death'].astype(int) > 365), 'event_death'] = 0
        print(f'events after 1yr:')
        print(ext_df['event_death'].sum())
        
    else:
        pass

  elif main_args.spec_event == 'prog':

    if main_args.spec_duration == '1yr':
        ext_df = ext_df.astype({'event_prog': 'int'})
        print('events before 1yr:')
        print(ext_df['event_prog'].sum())
        ext_df.loc[(ext_df['event_prog'] == 1) & (ext_df['duration_prog'].astype(int) > 365), 'event_prog'] = 0
        print(f'events after 1yr:')
        print(ext_df['event_prog'].sum())
        
    else:
        pass

  if args.net_architect =='VisionTransformer':
    ext_df_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_vit.csv')
    ext_df.to_csv(ext_df_path)
  
  elif args.net_architect == 'DenseNet' or 'resnet50_cbam' or 'SEResNext50':
    ext_df_path = os.path.join(args.label_dir, f'{main_args.ext_dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
    ext_df.to_csv(ext_df_path)
  
  print(f'saving new label csv file for external set {main_args.ext_dataset_name} at {ext_df_path}') # 

  return ext_df

#%%
def make_kfold_df_proc_labels(main_args, args, dataset_name, remove_idh_mut = False, fixed_interval_width = 0):
  
  if args.net_architect =='VisionTransformer':
    get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_vit.csv')
  elif args.net_architect =='DenseNet' or 'resnet50_cbam' or 'SEResNext50':  
    get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}.csv')
  
  get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                          np.array(df[f'duration_{main_args.spec_event}'].tolist(), dtype=int), 
                          np.array(df[f'event_{main_args.spec_event}'].tolist(), dtype=int))
  
  df = pd.read_csv(get_label_path(dataset_name), dtype='string')
  df = df.set_index('ID')
  df = df.sort_index(ascending=True)
  df = df[args.data_key_list]
  print(f'df.index.values:{len(df.index.values)}')
  
  if remove_idh_mut:
    condition = df.IDH.astype(int) == 0 # 1 means mut, not 0 
    filtered_ID = df[condition].index.tolist() 
    df = df.loc[sorted(filtered_ID),:]
    print(f'after removing idh mutation; df.index.values:{len(df.index.values)}')
  
  if '_' in dataset_name:
    list_dataset = dataset_name.split('_')
    print(f'list_dataset: {list_dataset}')
    
    comm_list = []
    for split_dataset in list_dataset:
      print(f'split_dataset: {split_dataset}')
      
      if args.net_architect =='VisionTransformer':
         img_dir = os.path.join(args.data_dir, split_dataset, "VIT",f'{args.compart_name}_BraTS_v2')
         split_comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
         print(f'split_img_label_comm_list:{len(split_comm_list)}')
         comm_list.extend(split_comm_list)
      
      elif args.net_architect =='DenseNet' or 'SEResNext50' or 'resnet50-cbam':
        img_dir = os.path.join(args.data_dir, split_dataset, f'{args.compart_name}_BraTS_v2')
        split_comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
        print(f'split_img_label_comm_list:{len(split_comm_list)}')
        comm_list.extend(split_comm_list)
      
  else:  
    if args.net_architect =="VisionTransformer":
       img_dir = os.path.join(args.data_dir, dataset_name, "VIT",f'{args.compart_name}_BraTS_v2')
       comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
       
    elif args.net_architect =='DenseNet' or 'SEResNext50' or 'resnet50-cbam':
       img_dir = os.path.join(args.data_dir, dataset_name, f'{args.compart_name}_BraTS_v2') 
       comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]

  print(f'img_label_comm_list:{len(comm_list)}')
  print(f'dataset_name:{dataset_name}, {len(comm_list)}') # SNUH_UPenn, 1113
  
  df = df.loc[sorted(comm_list)] #.astype(int) 

  print(f'{dataset_name} df.shape: {df.shape}') # (1113, 8) 

  ID, duration, event = get_target(df)
  
  kfold = add_kfold_to_df(df, args, main_args.seed)
  
  breaks, _ = get_n_intervals(fixed_interval_width)

  proc_labels = make_surv_array(duration, event, breaks)
  df_proc_labels = pd.DataFrame(proc_labels)

  df_proc_labels['ID'] = ID
  df_proc_labels['kfold'] = kfold
  df_proc_labels = df_proc_labels.set_index('ID')
  
  proc_label_path = os.path.join(args.proc_label_dir, f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_proc_labels.csv')
  df_proc_labels.to_csv(proc_label_path)
  
  return df_proc_labels, event, duration

#%%
def make_class_df(main_args,args, dataset_name, remove_idh_mut = False):
  
  get_label_path = lambda dataset: os.path.join(args.label_dir, f'{dataset}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_vit.csv')
  get_target = lambda df: (np.array(df.index.values, dtype=str), # int: not working
                           np.array(df["glioma_num"].tolist(), dtype=int))
  
  df = pd.read_csv(get_label_path(dataset_name), dtype='string')
  df = df.set_index('ID')
  df = df.sort_index(ascending=True)
  df = df[args.data_key_list]
  print(f'df.index.values:{len(df.index.values)}')

  if remove_idh_mut:
    condition = df.IDH.astype(int) == 0 # 1 means mut, not 0 
    filtered_ID = df[condition].index.tolist() 
    df = df.loc[sorted(filtered_ID),:]
    print(f'after removing idh mutation; df.index.values:{len(df.index.values)}')
  
  if '_' in dataset_name:
    list_dataset = dataset_name.split('_')
    print(f'list_dataset: {list_dataset}')
    
    comm_list = []
    for split_dataset in list_dataset:
      print(f'split_dataset: {split_dataset}')
      if args.net_architect =='VisionTransformer':
         img_dir = os.path.join(args.data_dir, split_dataset, "VIT",f'{args.compart_name}_BraTS_v2')
      else:
         img_dir = os.path.join(args.data_dir, split_dataset, f'{args.compart_name}_BraTS_v2')
      
      split_comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
      print(f'split_img_label_comm_list:{len(split_comm_list)}')
      comm_list.extend(split_comm_list)
    
  else:  
    img_dir = os.path.join(args.data_dir, dataset_name, f'{args.compart_name}_BraTS_v2') 
    comm_list = [elem for elem in get_dir(img_dir) if elem in df.index.values]
  
  print(f'img_label_comm_list:{len(comm_list)}')
  print(f'dataset_name:{dataset_name}, {len(comm_list)}') # SNUH_UPenn, 1113
  
  df = df.loc[sorted(comm_list)] #.astype(int) 

  print(f'{dataset_name} df.shape: {df.shape}') # (1113, 8) 

  ID, glioma_class = get_target(df)
  
  copy_df = df.copy()
  copy_df["ID"] = ID
  copy_df["glioma_num"] = glioma_class
  
  class_df = pd.concat([copy_df["ID"],copy_df["glioma_num"]],axis=1)
  class_df = class_df.set_index("ID")
  class_label_path = '/mnt/hdd3/mskim/GBL/data/label/class_labels'
  class_labels_path = os.path.join(class_label_path,f'{dataset_name}_{main_args.spec_duration}_{main_args.spec_patho}_{main_args.spec_event}_class_labels.csv')
  class_df.to_csv(class_labels_path)
  
  return class_df
  
#%%
def nnet_pred_surv(y_pred, breaks, fu_time):

  y_pred=np.cumprod(y_pred, axis=1)
  pred_surv = []
  for i in range(y_pred.shape[0]):
    pred_surv.append(np.interp(fu_time,breaks[1:],y_pred[i,:]))
  return np.array(pred_surv)

def add_kfold_to_df(df, args, seed):
  
  ''' Create Folds 
  ref:
  1) https://www.kaggle.com/code/debarshichanda/seresnext50-but-with-attention 
  2) https://stackoverflow.com/questions/60883696/k-fold-cross-validation-using-dataloaders-in-pytorch
  '''  
  
  skf = StratifiedKFold(n_splits=args.n_fold, shuffle=True, random_state=seed)
  for fold, ( _, val_) in enumerate(skf.split(X=df, y=df.event_death)):
      # print(fold, val_)
      # print(df.index[val_])
      df.loc[df.index[val_] , "kfold"] = int(fold)
      
  df['kfold'] = df['kfold'].astype(int)
  kfold = df['kfold'].values
  
  return kfold

def random_split(id_list, split_ratio):
  ''' df: dataframe for total dataset '''
  n_sample = len(id_list) 
  id_list = sorted(id_list)
  train_nums = np.random.choice(n_sample, size = int(split_ratio * n_sample), replace = False)
  print(f'train_nums:{len(train_nums)}')
  val_nums = [num for num in np.arange(n_sample) if num not in train_nums]
  
  return train_nums, val_nums

def make_surv_array(t,f,breaks):
  """Transforms censored survival data into vector format that can be used in Keras.
    Arguments
        t: Array of failure/censoring times.
        f: Censoring indicator. 1 if failed, 0 if censored.
        breaks: Locations of breaks between time intervals for discrete-time survival model (always includes 0)
    Returns
        Two-dimensional array of survival data, dimensions are number of individuals X number of time intervals*2
  """
  n_samples=t.shape[0]
  n_intervals=len(breaks)-1
  timegap = breaks[1:] - breaks[:-1]
  breaks_midpoint = breaks[:-1] + 0.5*timegap
  y_train = np.zeros((n_samples,n_intervals*2))
  for i in range(n_samples):
    if f[i]: #if failed (not censored)
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks[1:]) #give credit for surviving each time interval where failure time >= upper limit
      if t[i]<breaks[-1]: #if failure time is greater than end of last time interval, no time interval will have failure marked
        y_train[i,n_intervals+np.where(t[i]<breaks[1:])[0][0]]=1 #mark failure at first bin where survival time < upper break-point
    else: #if censored
      y_train[i,0:n_intervals] = 1.0*(t[i]>=breaks_midpoint) #if censored and lived more than half-way through interval, give credit for surviving the interval.
  return y_train

#%%

def fetch_scheduler(optimizer, args):
    if args.scheduler == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.min_lr)
    elif args.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=args.T_0, T_mult=1, eta_min=args.min_lr)
    elif args.scheduler == None:
        return None
        
    return scheduler

#%%



def run_fold(df, args, model, criterion, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold]
    train_df = df[df.kfold != fold]
    
    '''
    df_proc_labels_test 를 그냥 .csv 로 저장하고 load 하는 방식으로 하기
    '''
    train_df_path = os.path.join(args.proc_label_dir, f'train_df_proc_labels_{args.dataset_name}.csv')
    train_df.to_csv(train_df_path)

    valid_df_path = os.path.join(args.proc_label_dir, f'valid_df_proc_labels_{args.dataset_name}.csv')
    valid_df.to_csv(valid_df_path)
    
    train_data = SurvDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}', transforms=args.train_transform, aug_transform=True) #True)
    valid_data = SurvDataset(df = valid_df, args = args, dataset_name = f'{args.dataset_name}', transforms=args.valid_transform, aug_transform=False)
  
    dataset_sizes = {
        'train' : len(train_data),
        'valid' : len(valid_data)
    }
    
    print(f'num of train_data: {len(train_data)}')
    print(f'num of valid_data: {len(valid_data)}')
    
    train_loader = DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    
    dataloaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }

    model, history = train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold)
    
    return model, history

#%%
def run_fold_vit(df, args, model, criterion, optimizer, scheduler, device, fold, num_epochs=10):
    valid_df = df[df.kfold == fold]
    train_df = df[df.kfold != fold]

    train_data = ViTDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}')
    valid_data = ViTDataset(df = valid_df, args = args, dataset_name = f'{args.dataset_name}')
  
    dataset_sizes = {
        'train' : len(train_data),
        'valid' : len(valid_data)
    }
    
    print(f'num of train_data: {len(train_data)}')
    print(f'num of valid_data: {len(valid_data)}')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    
    valid_loader = torch.utils.data.DataLoader(dataset=valid_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=False)
    
    dataloaders = {
        'train' : train_loader,
        'valid' : valid_loader
    }

    model, history = train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold)
    
    return model, history

#%%
def train_classification(df, args, model, criterion, optimizer, scheduler, device, num_epochs=10):
    train_df = df

    train_data = ClassViTDataset(df = train_df, args = args, dataset_name = f'{args.dataset_name}')
    
    dataset_sizes = {
        'train' : len(train_data)
      }
    
    print(f'num of train_data: {len(train_data)}')
    
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, num_workers=4, pin_memory=True, shuffle=True)
    
    dataloaders = {
        'train' : train_loader
      }

    model, history = train_classification_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device)
    
    return model, history
#%%

def get_transform(args, dataset_name, train_option=False):
  
  if dataset_name == 'SNUH_UPenn_TCGA':
    landmark_dataset_name = 'SNUH_UPenn_TCGA' # train/valid/test=0.75/0.72/0.69 # 
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  else:
    landmark_dataset_name = dataset_name
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  
  landmarks_dir = os.path.join(args.data_dir, 'histograms', landmark_dataset_name)
  
  landmarks = {}
  for seq in args.sequence:
    
    seq_landmarks_path = os.path.join(landmarks_dir, f'{seq}_histgram.npy') if train_option else os.path.join(landmarks_dir, f'{seq}_histgram_train.npy')
    landmarks[f'{seq}'] = seq_landmarks_path
  
    
  basic_transforms = [
      tio.HistogramStandardization(landmarks), 
      # tio.ZNormalization() # (masking_method=lambda x: x > 0) # x.mean() # # NOT working: RuntimeError: Standard deviation is 0 for masked values in image    
  ]

  basic_transform = tio.Compose(basic_transforms)
  # aug_transform = Compose(aug_transforms)
  
  print(f'transform for {dataset_name} was obtained')
  
  return basic_transform

#%%
def get_transform_vit(args, dataset_name, train_option=False):
  
  if dataset_name == 'SNUH_UPenn_TCGA':
    landmark_dataset_name = 'SNUH_UPenn_TCGA' # train/valid/test=0.75/0.72/0.69 # 
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  else:
    landmark_dataset_name = dataset_name
    print(f'landmark_dataset_name:{landmark_dataset_name}')
  
  landmarks_dir = os.path.join(args.data_dir, 'histograms', landmark_dataset_name,"VIT") 
  
  landmarks = {}
  for seq in args.sequence:
    
    seq_landmarks_path = os.path.join(landmarks_dir, f'{seq}_histgram.npy') if train_option else os.path.join(landmarks_dir, f'{seq}_histgram_train.npy')
    landmarks[f'{seq}'] = seq_landmarks_path
  basic_transforms = [
      tio.HistogramStandardization(landmarks), 
      # tio.ZNormalization() # (masking_method=lambda x: x > 0) # x.mean() # # NOT working: RuntimeError: Standard deviation is 0 for masked values in image    
  ]

  basic_transform = tio.Compose(basic_transforms)
  # aug_transform = Compose(aug_transforms)
  
  print(f'transform for {dataset_name} was obtained')
  
  return basic_transform

#%%
def load_ckpt(args, model):
  ckpt_dir = os.path.join(os.getcwd(), 'saved_models', f'{args.net_architect}')
  os.makedirs(ckpt_dir, exist_ok = True)
  ckpt_list = glob.glob(f'{ckpt_dir}/*.pth') 
  ckpt_model = max(ckpt_list, key=os.path.getctime)
  print(f'latest_ckpt_model: {ckpt_model}') #'Fold0_3.1244475595619647_epoch23.pth'
  ckpt_path = os.path.join(ckpt_dir, ckpt_model) 
  model_dict = torch.load(ckpt_path, map_location='cuda') # NOT working when in utils.py: f'cuda:{gpu_id}'
  model.load_state_dict(model_dict)
  return model

def load_model(args, model):
  ckpt_dir = os.path.join(args.pth_path, 'saved_models', f'{args.net_architect}',f'{args.experiment_mode}')
  # os.makedirs(ckpt_dir, exist_ok = True)
  ckpt_list = glob.glob(f'{ckpt_dir}/*.pth') 
  ckpt_model = max(ckpt_list, key=os.path.getctime)
  print(f'latest_ckpt_model: {ckpt_model}') #'Fold0_3.1244475595619647_epoch23.pth'
  ckpt_path = os.path.join(ckpt_dir, ckpt_model) 
  model = torch.load(ckpt_path, map_location='cuda') # NOT working when in utils.py: f'cuda:{gpu_id}'
  
  return model

class SurvDataset(nn.Module):
  def __init__(self, df, args, dataset_name, transforms=None, aug_transform=False): # ['27179925', '45163562', 'UPENN-GBM-00291_11', '42488471', 'UPENN-GBM-00410_11', '28802482']
    self.dataset_name = dataset_name
    self.df = df 
    self.args = args
    # print(self.df.shape) # (890, 39) # (223, 39)
    self.img_dir = os.path.join(args.data_dir, self.dataset_name, f'{args.compart_name}_BraTS_v2') # 'SNUH_UPenn_TCGA_severance'
    self.transforms = transforms
    self.aug_transform = aug_transform

    self.znorm = tio.ZNormalization()
    self.rescale = tio.RescaleIntensity(out_min_max=(-1, 1))
    self.crop_size = 64
    self.crop = RandSpatialCrop(roi_size=(self.crop_size, self.crop_size, self.crop_size), random_size=False)
    
    # self.rand_affiner = RandAffine(prob=0.9, rotate_range=[-0.5,0.5], translate_range=[-7,7],scale_range= [-0.15,0.1], padding_mode='zeros')
    self.rand_affiner = RandAffine(prob=0.9)
    self.rand_elastic = Rand3DElastic(prob=0.8, magnitude_range = [-1,1], sigma_range = [0,1])
    self.flipper1 = RandFlip(prob=0.5, spatial_axis=0)
    self.flipper2 = RandFlip(prob=0.5, spatial_axis=1)
    self.flipper3 = RandFlip(prob=0.5, spatial_axis=2)
    self.gaussian = RandGaussianNoise(prob=0.3)
    self.contrast = AdjustContrast(gamma=2)
    self.compart_name = args.compart_name
  
  def concat_seq_img(self, x):
      return torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], axis=0)
  
  def __len__(self):
    return len(self.df) # 결국 df 로 index 를 하기 때문에 dataset의 길이도 len(df): df를 train_df, val_df 넣는것에 따라 dataset이 train_set, val_set이 됨.

  def augment(self, img):
    img = self.crop(img)
    img = self.gaussian(img)
    
    return img

  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = self.df.iloc[idx].name
    kfold = self.df['kfold'][idx]
        
    subj_img_dir = os.path.join(self.img_dir, str(ID))
            
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')), # t1_seg.nii.gz
        t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
        t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
        flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz')), 
            
        )   
    
    if self.transforms:
      subject = self.transforms(subject)
    
    img = self.concat_seq_img(subject)
    # print(f'img loaded: {img.shape}') # torch.Size([4, 12])
    
    if self.aug_transform:
      img = self.augment(img)
    # print(f'final input image shape: {img.shape}') # torch.Size([4, 120, 120, 78])
    
    proc_label_list = list(self.df[self.df.columns.difference(['ID', 'kfold'])].iloc[idx].values) # ID, kfold 는 제외
    proc_labels = [int(float(proc_label)) for proc_label in proc_label_list] # '1.0' -> 1: string -> float -> int
    proc_labels = torch.tensor(proc_labels)
    # print(f'proc_labels.shape:{proc_labels.shape}') # torch.Size([38])
    
    return img, proc_labels

class ViTDataset(nn.Module):
  def __init__(self, df, args, dataset_name):
    self.dataset_name = dataset_name
    self.df = df 
    self.args = args
    self.img_dir = os.path.join(args.data_dir, self.dataset_name,"VIT", f'{args.compart_name}_BraTS_v2') # 'SNUH_UPenn_TCGA_severance'
    self.compart_name = args.compart_name
    
  def concat_seq_img(self, x):
      seq_cat = torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], dim=0)
      return seq_cat.to(torch.float32)
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = self.df.iloc[idx].name
    kfold = self.df['kfold'][idx]
        
    subj_img_dir = os.path.join(self.img_dir, str(ID))
            
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')),
        t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
        t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
        flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz')),
        )

    img = self.concat_seq_img(subject)
    # print(f'img loaded: {img.shape}') -> torch(4,224,224,224)
    
    proc_label_list = list(self.df[self.df.columns.difference(['ID', 'kfold'])].iloc[idx].values) # ID, kfold 는 제외
    proc_labels = [int(float(proc_label)) for proc_label in proc_label_list] # '1.0' -> 1: string -> float -> int
    proc_labels = torch.tensor(proc_labels)
    # print(f'proc_labels.shape:{proc_labels.shape}') torch(38)
    return img, proc_labels

class ClassViTDataset(nn.Module):
  def __init__(self, df, args, dataset_name):
    self.dataset_name = dataset_name
    self.df = df 
    self.args = args
    self.img_dir = os.path.join(args.data_dir, self.dataset_name,"VIT", f'{args.compart_name}_BraTS_v2') # 'SNUH_UPenn_TCGA_severance'
    self.compart_name = args.compart_name
    
  def concat_seq_img(self, x):
      seq_cat = torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], dim=0)
      return seq_cat.to(torch.float32)
  
  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = self.df.iloc[idx].name
        
    subj_img_dir = os.path.join(self.img_dir, str(ID))
            
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')),
        t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
        t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
        flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz')),
        )

    img = self.concat_seq_img(subject)
    
    glioma_class_num = self.df["glioma_num"].iloc[idx]
    class_label = int(glioma_class_num) # wild : 1 , astrocytoma : 2 , oligodendro : 3, mutant : 4, gliosarcoma : 5 (only severance)
    class_label = torch.tensor(class_label-1, dtype=torch.float32) # wild : 0 , astrocytoma : 1 , oligodendro : 2, mutant : 3, gliosarcoma : 4 (only severance)
    
    return img, class_label

#%%
class PropHazards(nn.Module):
  def __init__(self, size_in, size_out):#, device):
    super().__init__()
    self.linear = nn.Linear(size_in, size_out)#.to(device)

  def forward(self, x):
    # print('self.linear.weight:', self.linear.weight)
    
    x = self.linear(x)
    # if np.any(np.isnan(x)):
    #   pdb.set_trace()
    # print('torch.sigmoid(x):',torch.sigmoid(x))
    # print('torch.exp(x):',torch.exp(x))
    
    return torch.pow(torch.sigmoid(x), torch.exp(x)) #.float().to(device)

class Hazards(nn.Module):
  def __init__(self, size_in, size_out):#, device):
    super().__init__()
    self.linear = nn.Linear(size_in, size_out)#.to(device)

  def forward(self, x):
    x = self.linear(x)
    
    return torch.sigmoid(x) #.float().to(device)

class MutilLayer(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, n_layers, drop_rate=0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=drop_rate))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, input_dim))
        layers.append(nn.GELU())

        super().__init__(*layers)

class MutilLayer_Outdim(nn.Sequential):
    def __init__(self, input_dim, hidden_dim, out_dim, n_layers, drop_rate=0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=drop_rate))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.GELU())

        super().__init__(*layers)

class HazardNetwork(nn.Module):
    def __init__(self, args,
                 hidden_dim,
                 num_category, 
                #  batch_size,
                 ):
        super(HazardNetwork, self).__init__()
        
        self.args=args
        # self.encoder = encoder
        if args.hazard_layer_depth == 0:
          self.hazard_network = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())
        else:
          self.hazard_network = nn.Sequential(*[MutilLayer(hidden_dim, hidden_dim*self.args.hazard_exdim, self.args.hazard_layer_depth, self.args.drop_rate),nn.Linear(hidden_dim, 1), nn.Sigmoid()])
        self.num_category = num_category
        
        # self.batch_size = batch_size

    def forward(self, representation, batch_size):
        # pdb.set_trace()
        # representation = self.encoder(x)
        # batch_size=int(representation.shape[0]/self.num_category) #input은 이미 expand된 형태
        times = ((self.min_max_normalization(torch.tensor(np.arange(0, self.num_category)))).unsqueeze(0).expand(batch_size, 1, self.num_category).reshape(-1,1)).to(self.args.device)

        representation = torch.concat((representation, times),dim=1)
        hazard = self.hazard_network(representation)
        return hazard
    
    def min_max_normalization(self, tensor):
        min_value = torch.min(tensor)
        max_value = torch.max(tensor)

        normalized_tensor = (tensor - min_value) / (max_value - min_value)

        return normalized_tensor

class CustomNetwork(nn.Module):
  def __init__(self, args, base_model):
    super().__init__()
    self.base_model = base_model
    self.model = vars(base_model)['_modules'] #["_modules"] = nn.Module 상속받고 모델 구조만 확인하고 싶을 때, vars = Dictionary
    
    self.output_dim = args.n_intervals # 19
    print(f'self.output_dim:{self.output_dim}')
    layers = []
    self.layer_name_list = [n for n in self.model][:-1]
    print(f'self.layer_name_list:{self.layer_name_list}')

    for name in self.layer_name_list:
       layers.append(self.model[name])
      
    if args.net_architect == 'DenseNet':
       layers.append(self.model['class_layers'][:2])

    if args.net_architect == 'SEResNext50':
       self.num_out = 2048
    elif args.net_architect == 'DenseNet':
       self.num_out = 1024
    elif args.net_architect == 'resnet50_cbam':
       self.num_out = 2048
    elif args.net_architect == 'VisionTransformer':
       self.num_out = 1024

    
    self.layer1 = nn.ModuleList(layers)
    self.flatten = nn.Flatten()
    self.prophazards = PropHazards(self.num_out, self.output_dim) # (size_in = args.last_size[args.net_architect], size_out = self.output_dim)
    
  def forward(self, x):
    for layer in self.layer1:
       x = layer(x)
    # print(f'x.size:{x.size()}')

    x = self.flatten(x)
    x = self.prophazards(x)
    
    return x

class CustomNetwork_edit(nn.Module):
  def __init__(self, args, base_model):
    super().__init__()
    self.base_model = base_model
    layer_names = []
    for name, module in base_model.named_modules():
        layer_names.append(name)

    # self.output_dim = args.n_intervals # 19
    self.args=args
    self.num_Event = 1
      
    # if args.net_architect == 'DenseNet':
    #    layers.append(self.model['class_layers'][:2])

    if args.net_architect == 'SEResNext50':
       self.base_model._modules[layer_names[-1]] = nn.Identity()
       self.num_out = 2048
    elif args.net_architect == 'DenseNet':
       self.base_model.class_layers.flatten= nn.Identity()
       self.base_model.class_layers.out= nn.Identity()
       self.num_out = 1024
    elif args.net_architect == 'resnet50_cbam':
       self.base_model._modules[layer_names[-1]] = nn.Identity()
       self.num_out = 2048
    elif args.net_architect == 'VisionTransformer':
       self.base_model._modules[layer_names[-1]] = nn.Identity()
       self.num_out = 1024
    elif args.net_architect == 'pretrained_resnet':
       self.base_model._modules[layer_names[-1]] = nn.Identity()
       self.num_out = 2048
    elif args.net_architect == 'DINO':
       self.num_out = 768
    
    # self.layer1 = nn.ModuleList(layers)
    # self.flatten = nn.Flatten()
    # self.prophazards = PropHazards(self.num_out, self.args.num_category) # (size_in = args.last_size[args.net_architect], size_out = self.output_dim)
    # self.hazards = Hazards(self.num_out, self.args.num_category)
    self.hazards_net=HazardNetwork(self.args, self.num_out+1, self.args.num_category) # len([1,3,6,12,24,36,48,60]) =8

    # self.o_layer = nn.Sequential(
    #         nn.Linear(self.num_out, self.args.num_category),
    #         nn.Softmax(dim=1)
    #     )
    
  def forward(self, x):
    # for layer in self.layer1:
    #    x = layer(x)
    # print(f'x.size:{x.size()}')
    # # pdb.set_trace()



    # x= self.base_model(x)

    # # x = self.flatten(x)

    # # x = self.prophazards(x)
    # x=self.hazards(x)

    
    # # for risk function
    # # x = x.reshape(-1, self.num_Event * self.num_out)
    # # OUTPUT layers
    # # x = self.o_layer(x)
    # x = x.reshape(-1, self.args.num_category)

    batch_size=x.shape[0]
    # x=torch.unsqueeze(x,1).expand(x.shape[0],self.args.num_category, x.shape[1], x.shape[2], x.shape[3],x.shape[4]).reshape(-1, x.shape[1], x.shape[2], x.shape[3],x.shape[4]).to(self.args.device)
    # (B,C,W,H) -> (B,1,C,W,H) -> (B,num_category,C,W,H) -> (B*num_category,C,W,H)
    x= self.base_model(x)
    # x.shape: (B*num_category, num_out)

    # x = self.flatten(x)
    x=torch.unsqueeze(x,1).expand(x.shape[0],self.args.num_category, x.shape[1]).reshape(-1, x.shape[1]).to(self.args.device)
    # x = self.prophazards(x)
    x=self.hazards_net(x,batch_size)

    x=x.reshape(batch_size, self.args.num_category)
    # for risk function
    # x = x.reshape(-1, self.num_Event * self.num_out)
    # OUTPUT layers
    # x = self.o_layer(x)
    # x = x.reshape(-1, self.args.num_category)


    return x


class CustomNetwork_edit_for_partition(nn.Module):
  def __init__(self, args, base_model, partition_cfg):
    super().__init__()
    self.base_model = base_model
    self.partition_cfg=partition_cfg
    layer_names = []
    for name, module in base_model.named_modules():
        layer_names.append(name)

    # self.output_dim = args.n_intervals # 19
    self.args=args
    self.num_Event = 1
      
    # if args.net_architect == 'DenseNet':
    #    layers.append(self.model['class_layers'][:2])

    if args.net_architect == 'SEResNext50':
       self.base_model._modules[layer_names[-1]] = nn.Identity()
       self.num_out = 2048
    elif args.net_architect == 'DenseNet':
       self.base_model.class_layers.flatten= nn.Identity()
       self.base_model.class_layers.out= nn.Identity()
       self.num_out = 1024
    elif args.net_architect == 'resnet50_cbam':
       self.base_model._modules[layer_names[-1]] = nn.Identity()
       self.num_out = 2048
    elif args.net_architect == 'VisionTransformer':
       self.base_model._modules[layer_names[-1]] = nn.Identity()
       self.num_out = 1024

    
    # self.layer1 = nn.ModuleList(layers)
    self.flatten = nn.Flatten()
    # self.prophazards = PropHazards(self.num_out, self.args.num_category) # (size_in = args.last_size[args.net_architect], size_out = self.output_dim)
    self.hazards = Hazards(self.num_out, self.args.num_category)
    # self.o_layer = nn.Sequential(
    #         nn.Linear(self.num_out, self.args.num_category),
    #         nn.Softmax(dim=1)
    #     )
    
    self.encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8).to(self.args.device)
    self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6).to(self.args.device)
    self.permute_layer=ChannelAdjustment(self.partition_cfg.partition_num,1)
    

  def forward(self, x):
    # for layer in self.layer1:
    #    x = layer(x)
    # print(f'x.size:{x.size()}')


    
    x= self.base_model(x)

    x=x.reshape(-1,self.partition_cfg.partition_num,self.num_out)
    
    #transformer encoder 추가
    x=self.transformer_encoder(x)
    # x1=x1.mean(dim=1)
    x=x.mean(dim=1)
    # x=self.permute_layer(x)
    x = self.flatten(x)

    # x3 = self.prophazards(x2)
    x=self.hazards(x)

    
    # for risk function
    # x = x.reshape(-1, self.num_Event * self.num_out)
    # OUTPUT layers
    # x = self.o_layer(x)
    x = x.reshape(-1, self.args.num_category)


    return x

class Custom_ResNet(nn.Module):
    def __init__(self,num_classes):
        super().__init__()
        self.num_classes=num_classes
        self.base_model=pretrained_ResNet()
        self.avgpool = nn.AdaptiveAvgPool3d(1) 
        self.fc= nn.Linear(512 * 4, num_classes)

    def forward(self,x):
        x=self.base_model(x)
        # pdb.set_trace()
        x=self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        #pdb.set_trace()
        x=self.fc(x)

        return x

def pretrained_ResNet():
    model = resnet.resnet50(
                    sample_input_W=448,
                    sample_input_H=448,
                    sample_input_D=56,
                    shortcut_type='B',
                    no_cuda=False,
                    num_seg_classes=2)
    net_dict = model.state_dict()
    pretrain = torch.load('./pretrain/resnet_50_23dataset.pth')
    pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        
    net_dict.update(pretrain_dict)
    model.load_state_dict(net_dict)
    model.conv1 = nn.Conv3d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv_seg = nn.Identity()



    return model

def get_output_shape(x, model):
  model.eval()
  x = model(x)
  return torch.tensor(x.shape[1:]).prod()#.cuda()

#%%
''' Loss Function '''

def cox_partial_likelihood(y_pred, y_true, n_intervals=19):
    # Get the number of samples
    num_samples = y_pred.size(0)
    
    # Extracting censoring and event information from y_true
    censored = 1. + y_true[:, 0:n_intervals] * (y_pred-1.)  # Censoring indicators
    events = 1. - y_true[:, n_intervals:2*n_intervals] * y_pred  # Event indicators
    
    # Calculating the risk scores
    risk_scores = torch.exp(y_pred)  # Using exponential as risk scores
    
    # Computing the log partial likelihood
    log_partial_likelihood = torch.zeros(num_samples, device=y_pred.device)
    for i in range(num_samples):
        risk_i = risk_scores[i]
        censored_i = censored[i]
        events_i = events[i]
        
        risk_sum = torch.sum(risk_scores[i:])
        events_sum = torch.sum(events[i:] * risk_scores[i:])
        
        log_partial_likelihood[i] = torch.log(events_sum / risk_sum)
    
    # Calculating the negative of the mean log partial likelihood
    loss = -torch.mean(log_partial_likelihood)
    
    return loss
  
def nnet_loss(y_pred, y_true, n_intervals = 19):
    
    ''' criterion argument의 순서 (y_pred, y_true 여야 돌아가고, y_true, y_pred 면 차원 안 맞음)가 중요하고 
    보통 (output, target 또는 label) 순으로 선언되며, 이 경우는 utils.py의 train_model 에 criterion(output, label) 로 되어 있음. '''
    
    cens_uncens = 1. + y_true[:, 0:n_intervals] * (y_pred-1.)
    
    uncens = 1. - y_true[:, n_intervals: 2 * n_intervals] * y_pred
    
    # print(f'y_pred.size:{y_pred.size()}') torch.size(10,19)
    # print(f'y_true.size:{y_true.size()}') torch.size(10,38)
    # print(f'cens_uncens:{cens_uncens.size()}') torch.size(10,19)
    # print(f'uncens.size:{uncens.size()}') torch.size(10,19)

    loss = torch.sum(-torch.log(torch.clip(torch.cat((cens_uncens, uncens), dim=-1), torch.finfo(torch.float32).eps, None)), axis=-1)
    # print(f'loss.size:{loss.size()}') # torch.size(10)
    loss = loss.mean()
    
    return loss

def loss_Log_Likelihood( out, k, fc_mask1):
    I_1 = torch.sign(k)

    #for uncensored: log P(T=t,K=k|x)
    tmp1 = torch.sum(torch.sum(fc_mask1 * out, dim=2), dim=1, keepdims=True)
    tmp1 = I_1 * log_(tmp1)

    #for censored: log \sum P(T>t|x)
    tmp2 = torch.sum(torch.sum(fc_mask1 * out, dim=2), dim=1, keepdims=True)
    tmp2 = (1. - I_1) * log_(tmp2)

    return - torch.mean(tmp1 + tmp2)

### LOSS-FUNCTION 2 -- Ranking loss
def loss_Ranking(out, k, t, fc_mask2, num_Category):
    sigma1 = 0.1
    eta = []
    for e in range(1):
        one_vector = torch.ones_like(t)
        e = torch.tensor(e, dtype=torch.int)
        I_2 = torch.eq(k, e+1) #indicator for event
        #I_2 = I_2.clone().detach().requires_grad_(True)
        I_2 = torch.tensor(I_2, dtype=torch.float32)
        I_2 = torch.diag(torch.squeeze(I_2))

        out_e = out[:, e:e+1, :] #각 event별로 따로 받아와서 계산
        tmp_e = out_e.reshape(-1, num_Category) #event specific joint prob.

        R = torch.matmul(tmp_e, fc_mask2.transpose(0,1)) #no need to divide by each individual dominator
        # r_{ij} = risk of i-th pat based on j-th time-condition (last meas. time ~ event time) , i.e. r_i(T_{j})

        diag_R = torch.reshape(torch.diagonal(R), [-1, 1])
        R = torch.matmul(one_vector, diag_R.transpose(0,1)) - R # R_{ij} = r_{j}(T_{j}) - r_{i}(T_{j})
        R = R.transpose(0,1)                                 # Now, R_{ij} (i-th row j-th column) = r_{i}(T_{i}) - r_{j}(T_{i})

        T = torch.sign(torch.matmul(one_vector, t.transpose(0,1)) - torch.matmul(t, one_vector.transpose(0,1)))
        relu = nn.ReLU()
        T = relu(T)
        # T_{ij}=1 if t_i < t_j  and T_{ij}=0 if t_i >= t_j

        T = torch.matmul(I_2, T) # only remains T_{ij}=1 when event occured for subject i
        tmp_eta = torch.mean(T * torch.exp(-R/sigma1), dim=1, keepdim=True)

        eta.append(tmp_eta)

    eta = torch.stack(eta, dim=1) #stack referenced on subjects
    eta = torch.mean(eta.reshape(-1, 1), dim=1, keepdim=True)

    return torch.sum(eta) #sum over num_Events

def uncensor( h, mask_1, mask_2, label):
    # h(t)
    p1 = log_(h) * mask_1
    p1 = p1.sum(dim=-1)

    # (1-h(t))
    p2 = log_(1-h) * mask_2
    p2 = p2.sum(dim=-1)

    # neg log likelihood
    p = p1 + p2

    # indicator
    p = p * label.squeeze()
    p = -(p.mean(dim=-1))

    return p

def censor(h, mask2, label):
    # (1-h(t))
    p = log_(1-h) * mask2
    p = p.sum(dim=-1)
    
    p = p * (1-(label.squeeze()))
    p = -(p.mean(dim=-1))

    return p

def nll_loss(h, mask_1, mask_2, label):
  uncensor_loss=uncensor(h, mask_1, mask_2, label)
  censor_loss=censor(h, mask_2, label)
  return uncensor_loss+censor_loss

def rank(hazard, mask_3, time, label, sigma=0.5):
    # Ranking Loss 1
    hazard = hazard.float()
    mask_3 = mask_3.float()
    time = time.float()
    label = label.float()

    H = hazard.cumsum(1)
    F = 1-torch.exp(-H)

    ones = torch.ones_like(time)
    I = torch.tensor(torch.eq(label, 1)).float()
    I = torch.diag(torch.squeeze(I))
    
    R = torch.matmul(F, mask_3.transpose(0,1))
    diag_R = R.diag().view(1,-1)
    R = ones.matmul(diag_R) - R
    R = R.transpose(0,1)

    T = torch.nn.functional.relu(torch.sign(torch.matmul(ones, time.transpose(0,1)) - torch.matmul(time, ones.transpose(0,1))))
    T = torch.matmul(I,T)

    temp = torch.mean(T * torch.exp(-R/sigma), dim=1, keepdim=True)
    temp = torch.mean(temp)

    return temp

def log_(x):
    return torch.log(x + 1e-8)

class TaylorSoftmax(nn.Module):
    def __init__(self, dim=1, n=2):
        super(TaylorSoftmax, self).__init__()
        assert n % 2 == 0
        self.dim = dim
        self.n = n

    def forward(self, x):
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.n+1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out

class LabelSmoothingLoss(nn.Module): 
    def __init__(self, classes=5, smoothing=0.0, dim=-1): 
        super(LabelSmoothingLoss, self).__init__() 
        self.confidence = 1.0 - smoothing 
        self.smoothing = smoothing 
        self.cls = classes 
        self.dim = dim 
    def forward(self, pred, target):
        with torch.no_grad():
            true_dist = torch.zeros_like(pred) # Real Data Distribution
            true_dist.fill_(self.smoothing / (self.cls - 1)) 
            target = target.data.unsqueeze(1).long() # Real Data Value
            true_dist.scatter_(1, target, self.confidence) 
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))
    

class TaylorCrossEntropyLoss(nn.Module):
    def __init__(self, n=2, ignore_index=-1, reduction='mean', smoothing=0.2):
        super(TaylorCrossEntropyLoss, self).__init__()
        assert n % 2 == 0
        self.taylor_softmax = TaylorSoftmax(dim=1, n=n)
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.lab_smooth = LabelSmoothingLoss(5, smoothing=smoothing)

    def forward(self, logits, labels):
        log_probs = self.taylor_softmax(logits).log()
        loss = self.lab_smooth(log_probs, labels)
        return loss

#%%
''' Training Function '''

def train_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device, fold):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_loss = 100
    history = defaultdict(list)
    model = model.to(device)

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train','valid']:
            if(phase == 'train'):
                print(f'phase:{phase}')
                model.train() # Set model to training mode
            else:
                print(f'phase:{phase}')
                model.eval() # Set model to evaluation mode
            
            running_loss = 0.0
            # running_corrects = 0.0
            
            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                 
                    loss = criterion(outputs, labels) # use this loss for any training statistics
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # first forward-backward pass
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        # second forward-backward pass
                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)


                running_loss += loss.item()*inputs.size(0)
            
            epoch_loss = running_loss/dataset_sizes[phase]
            # epoch_acc = running_corrects/dataset_sizes[phase]

            history[phase + ' loss'].append(epoch_loss)
            # history[phase + ' acc'].append(epoch_acc)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            # deep copy the model
            if phase=='valid' and epoch_loss <= best_loss: # epoch_acc >= best_acc:
                # best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                saved_model_path = os.path.join(os.getcwd(), 'saved_models', f'{args.net_architect}')
                os.makedirs(saved_model_path, exist_ok=True)
                PATH = os.path.join(saved_model_path, f"{datetime.now().strftime('%d_%m_%H_%m')}_{args.net_architect}_epoch{epoch}.pth")
                torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    # print("Best Accuracy ",best_acc)
    print("Best Loss ",best_loss)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history
  
''' Training Function '''

def train_classification_model(args, model, criterion, optimizer, scheduler, num_epochs, dataloaders, dataset_sizes, device,):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    # best_acc = 0.0
    best_loss = 100
    history = defaultdict(list)
    model = model.to(device)

    for epoch in range(1,num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train']:
            if(phase == 'train'):
                print(f'phase:{phase}')
                model.train() # Set model to training mode
            else:
                print(f'phase:{phase}')
                model.eval() # Set model to evaluation mode
            
            running_loss = 0.0
            # running_corrects = 0.0
            
            # Iterate over data
            for inputs,labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                 
                    loss = criterion(outputs, labels) # use this loss for any training statistics
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # first forward-backward pass
                        loss.backward()
                        optimizer.first_step(zero_grad=True)
                        
                        # second forward-backward pass
                        criterion(model(inputs), labels).backward()
                        optimizer.second_step(zero_grad=True)


                running_loss += loss.item()*inputs.size(0)
            
            epoch_loss = running_loss/dataset_sizes[phase]
            # epoch_acc = running_corrects/dataset_sizes[phase]

            history[phase + ' loss'].append(epoch_loss)
            # history[phase + ' acc'].append(epoch_acc)

            if phase == 'train' and scheduler != None:
                scheduler.step()

            print('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            
            # deep copy the model
            if epoch_loss <= best_loss: # epoch_acc >= best_acc:
              # best_acc = epoch_acc
              best_loss = epoch_loss
              best_model_wts = copy.deepcopy(model.state_dict())
              saved_model_path = os.path.join(os.getcwd(), 'saved_models', f'{args.net_architect}')
              os.makedirs(saved_model_path, exist_ok=True)
              PATH = os.path.join(saved_model_path, f"{datetime.now().strftime('%d_%m_%H_%m')}_{args.net_architect}_epoch{epoch}.pth")
              torch.save(model.state_dict(), PATH)

        print()

    end = time.time()
    time_elapsed = end - start
    print('Training complete in {:.0f}h {:.0f}m {:.0f}s'.format(
        time_elapsed // 3600, (time_elapsed % 3600) // 60, (time_elapsed % 3600) % 60))
    # print("Best Accuracy ",best_acc)
    print("Best Loss ",best_loss)

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

#%%

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                e_w = p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
                self.state[p]["e_w"] = e_w

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.sub_(self.state[p]["e_w"])  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        p.grad.norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
               )
        return norm

class FeatureExtractor():
    """ Class for extracting activations and
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                outputs += [x]
        return outputs, x

class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations = []
        for name, module in self.model._modules.items():
            if module == self.feature_module:
                target_activations, x = self.feature_extractor(x)
            elif "avgpool" in name.lower():
                x = module(x)
                x = x.view(x.size(0),-1)
            else:
                x = module(x)

        return target_activations, x

def preprocess_image(img):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    preprocessing = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    return preprocessing(img.copy()).unsqueeze(0)

def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        features, output = self.extractor(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()
        
        one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, input_img.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):
    @staticmethod
    def forward(self, input_img):
        positive_mask = (input_img > 0).type_as(input_img)
        output = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), input_img, positive_mask)
        self.save_for_backward(input_img, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input_img, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input_img > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input_img.size()).type_as(input_img),
                                   torch.addcmul(torch.zeros(input_img.size()).type_as(input_img), grad_output,
                                                 positive_mask_1), positive_mask_2)
        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        if self.cuda:
            input_img = input_img.cuda()

        input_img = input_img.requires_grad_(True)

        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = one_hot.cuda()

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)

        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)

# BS at 1yr
def get_BS(event, duration, oneyr_survs, duration_set=365):
  y = [(evt, dur) for evt, dur in zip(np.asarray(event, dtype=bool), duration)]
  y = np.asarray(y, dtype=[('cens', bool), ('time', float)])
  times, score = brier_score(y, y, oneyr_survs, duration_set)
  print(f'BS score at {duration_set}:{score}')
  return score

class ChannelAdjustment(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ChannelAdjustment, self).__init__()
        self.adjustment_layer = nn.Linear(input_channels, output_channels)

    def forward(self, x):
        # 입력 텐서의 각 채널을 새로운 가중치로 선형 변환
        output = self.adjustment_layer(x.permute(0, 2, 1))  # 입력 텐서를 (batch_size, width, num_channels) 형태로 변환
        output = output.permute(0, 2, 1)  # 출력 텐서를 (batch_size, num_channels, width) 형태로 변환
        output = output.squeeze(1)  # 중간 차원을 제거하여 출력 텐서의 형태를 변경
        return output
    
# event_death
def K_fold(df, data_set_name, kfold_num, random_state=42):
    if not os.path.exists(f'./kfold_index/{data_set_name}_fold_{kfold_num-1}_index1.npy'):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for i, (train_idx, test_idx) in enumerate (kfold.split(df,df['event_death'])):
            os.makedirs('./kfold_index',exist_ok=True)
            np.save(f'./kfold_index/{data_set_name}_fold_{i}_index.npy',test_idx)
        index_ls=[]
    for i in range(kfold_num):
        globals()["test_idx_{}".format(i)] = np.load(f'./kfold_index/{data_set_name}_fold_{i}_index.npy')

        globals()["df_test_fold_{}".format(i)]=df.loc[globals()["test_idx_{}".format(i)]]

        test_idx_var_name = "test_idx_{}".format(i)  # 검증 세트 인덱스 변수 이름 생성
        df_train_fold_var_name = "df_train_fold_{}".format(i)  # 훈련 데이터 세트 변수 이름 생성
        test_idx = globals()[test_idx_var_name]  # 검증 세트 인덱스 가져오기
        globals()[df_train_fold_var_name] = df.loc[~df.index.isin(test_idx)]
        tr_x ,vl_x= train_test_split(df.loc[~df.index.isin(test_idx)],test_size=0.2, shuffle=True, random_state=random_state, stratify=df.loc[~df.index.isin(test_idx)]['event_death'])
        globals()["df_train_fold_{}".format(i)] = tr_x
        globals()["df_valid_fold_{}".format(i)] = vl_x
    return (df_train_fold_0, df_valid_fold_0, df_test_fold_0), (df_train_fold_1, df_valid_fold_1, df_test_fold_1), (df_train_fold_2, df_valid_fold_2, df_test_fold_2), (df_train_fold_3, df_valid_fold_3, df_test_fold_3), (df_train_fold_4, df_valid_fold_4, df_test_fold_4)

# event_death_GBM
def K_fold_GBM(df, data_set_name, kfold_num, random_state=42):
    if not os.path.exists(f'./kfold_index_GBM/{data_set_name}_fold_{kfold_num-1}_index1.npy'):
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for i, (train_idx, test_idx) in enumerate (kfold.split(df,df['event_death'])):
            os.makedirs('./kfold_index_GBM',exist_ok=True)
            np.save(f'./kfold_index_GBM/{data_set_name}_fold_{i}_index.npy',test_idx)
        index_ls=[]
    for i in range(kfold_num):
        globals()["test_idx_{}".format(i)] = np.load(f'./kfold_index_GBM/{data_set_name}_fold_{i}_index.npy')

        globals()["df_test_fold_{}".format(i)]=df.loc[globals()["test_idx_{}".format(i)]]

        test_idx_var_name = "test_idx_{}".format(i)  # 검증 세트 인덱스 변수 이름 생성
        df_train_fold_var_name = "df_train_fold_{}".format(i)  # 훈련 데이터 세트 변수 이름 생성
        test_idx = globals()[test_idx_var_name]  # 검증 세트 인덱스 가져오기
        globals()[df_train_fold_var_name] = df.loc[~df.index.isin(test_idx)]
        tr_x ,vl_x= train_test_split(df.loc[~df.index.isin(test_idx)],test_size=0.2, shuffle=True, random_state=random_state, stratify=df.loc[~df.index.isin(test_idx)]['event_death'])
        globals()["df_train_fold_{}".format(i)] = tr_x
        globals()["df_valid_fold_{}".format(i)] = vl_x
    return (df_train_fold_0, df_valid_fold_0, df_test_fold_0), (df_train_fold_1, df_valid_fold_1, df_test_fold_1), (df_train_fold_2, df_valid_fold_2, df_test_fold_2), (df_train_fold_3, df_valid_fold_3, df_test_fold_3), (df_train_fold_4, df_valid_fold_4, df_test_fold_4)


class Tabular_HazardNet(nn.Module):
  def __init__(self, args, tabular_layer, tabular_layer_out_dim):
    super().__init__()

    # self.output_dim = args.n_intervals # 19
    self.args=args
    self.num_Event = 1
      
    self.num_out = tabular_layer_out_dim
    
    # self.layer1 = nn.ModuleList(layers)
    self.tabular_layer = tabular_layer
    self.hazards_net=HazardNetwork(self.args, self.num_out+1, self.args.num_category) # len([1,3,6,12,24,36,48,60]) =8

    # self.o_layer = nn.Sequential(
    #         nn.Linear(self.num_out, self.args.num_category),
    #         nn.Softmax(dim=1)
    #     )
    
  def forward(self, x):
    batch_size=x.shape[0]

    x = self.tabular_layer(x)

    x = torch.unsqueeze(x,1).expand(x.shape[0],self.args.num_category, x.shape[1]).reshape(-1, x.shape[1]).to(self.args.device)

    x = self.hazards_net(x,batch_size)

    x = x.reshape(batch_size, self.args.num_category)
    # for risk function
    # x = x.reshape(-1, self.num_Event * self.num_out)
    # OUTPUT layers
    # x = self.o_layer(x)
    # x = x.reshape(-1, self.args.num_category)


    return x



class Image_Tabular_HazardNet(nn.Module):
  def __init__(self, args, base_model, tabular_layer, tabular_layer_out_dim):
    super().__init__()
    self.base_model = base_model
    layer_names = []
    for name, module in base_model.named_modules():
        layer_names.append(name)

    # self.output_dim = args.n_intervals # 19
    self.args=args
    self.num_Event = 1
      
    # if args.net_architect == 'DenseNet':
    #    layers.append(self.model['class_layers'][:2])

    if args.net_architect == 'SEResNext50':
      self.base_model._modules[layer_names[-1]] = nn.Identity()
      image_num_out = 2048
    elif args.net_architect == 'DenseNet':
      self.base_model.class_layers.flatten= nn.Identity()
      self.base_model.class_layers.out= nn.Identity()
      image_num_out = 1024
    elif args.net_architect == 'resnet50_cbam':
      self.base_model._modules[layer_names[-1]] = nn.Identity()
      image_num_out = 2048
    elif args.net_architect == 'VisionTransformer':
      self.base_model._modules[layer_names[-1]] = nn.Identity()
      image_num_out = 1024
    elif args.net_architect == 'pretrained_resnet':
      self.base_model._modules[layer_names[-1]] = nn.Identity()
      image_num_out = 2048
    elif args.net_architect == 'DINO':
      image_num_out = 768

    self.num_out = tabular_layer_out_dim + image_num_out
    self.tabular_layer = tabular_layer
    # self.layer1 = nn.ModuleList(layers)
    # self.flatten = nn.Flatten()
    # self.prophazards = PropHazards(self.num_out, self.args.num_category) # (size_in = args.last_size[args.net_architect], size_out = self.output_dim)
    # self.hazards = Hazards(self.num_out, self.args.num_category)
    self.hazards_net=HazardNetwork(self.args, self.num_out + 1, self.args.num_category) # len([1,3,6,12,24,36,48,60]) =8

    # self.o_layer = nn.Sequential(
    #         nn.Linear(self.num_out, self.args.num_category),
    #         nn.Softmax(dim=1)
    #     )
    
  def forward(self, tabular, image):
    # for layer in self.layer1:
    #    x = layer(x)
    # print(f'x.size:{x.size()}')
    # # pdb.set_trace()



    # x= self.base_model(x)

    # # x = self.flatten(x)

    # # x = self.prophazards(x)
    # x=self.hazards(x)

    
    # # for risk function
    # # x = x.reshape(-1, self.num_Event * self.num_out)
    # # OUTPUT layers
    # # x = self.o_layer(x)
    # x = x.reshape(-1, self.args.num_category)
    
    # pdb.set_trace()
    batch_size= tabular.shape[0]
    # x=torch.unsqueeze(x,1).expand(x.shape[0],self.args.num_category, x.shape[1], x.shape[2], x.shape[3],x.shape[4]).reshape(-1, x.shape[1], x.shape[2], x.shape[3],x.shape[4]).to(self.args.device)
    # (B,C,W,H) -> (B,1,C,W,H) -> (B,num_category,C,W,H) -> (B*num_category,C,W,H)
    image_feat = self.base_model(image)

    tabular_feat = self.tabular_layer(tabular)
    # x.shape: (B*num_category, num_out)

    # x = self.flatten(x)
    combined_feat = torch.cat((image_feat, tabular_feat), dim=1)
    # pdb.set_trace()
    combined_feat=torch.unsqueeze(combined_feat,1).expand(combined_feat.shape[0],self.args.num_category, combined_feat.shape[1]).reshape(-1, combined_feat.shape[1]).to(self.args.device)
    # x = self.prophazards(x)
    combined_feat=self.hazards_net(combined_feat,batch_size)

    combined_feat=combined_feat.reshape(batch_size, self.args.num_category)
    # for risk function
    # x = x.reshape(-1, self.num_Event * self.num_out)
    # OUTPUT layers
    # x = self.o_layer(x)
    # x = x.reshape(-1, self.args.num_category)


    return combined_feat
  
# Transformer 

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x
    
def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    
class Transformer(nn.Module):
    def __init__(self, input_embed_dim=384, output_embed_dim = 192,
                 depth=12, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., 
                 norm_layer=nn.LayerNorm,
                 L=192, D=128, K=1,
                 **kwargs):
        super().__init__()

        embed_dim = output_embed_dim
        self.blocks = nn.ModuleList([ 
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        # [expected] Used for transforming the visual feature to a meaningfull feature.
        # change to batch
        B, len, dim = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        return x
    
    def forward(self, x):
        x_stages = []
        # Dimension change & Token added
        x = self.prepare_tokens(x) # cls token 추가
        for idx_stage, blk in enumerate(self.blocks): # 멀티헤드 어펜션
            x = blk(x) 
            x_stages.append(x)
            # print(f"\n{x.grad_fn}")
            
        x_last_stage = self.norm(x)
        output = x_last_stage[:,0,:] # cls토큰의 임베딩

        return output 
    

    def get_last_selfattention(self, x): # attention weight 확인
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
    

# FT-Transformer https://github.com/yandex-research/rtdl-revisiting-models/blob/main/bin/ft_transformer.py
import math
import torch.nn.init as nn_init
import typing as ty
from torch import Tensor

class TabularTokenizer(nn.Module):
    def __init__(
        self,
        d_numerical: int,  # 수치형 피처의 개수
        categories: ty.Optional[ty.List[int]],  # 범주형 피처별 고유값 개수 리스트 (ex) [3, 4, 2]
        d_token: int = 768,  # 임베딩 차원을 768로 고정
        bias: bool = True,
    ) -> None:
        super().__init__()
        
        if categories is None:
            d_bias = d_numerical
            self.category_offsets = None
            self.category_embeddings = None
        else:
            d_bias = d_numerical + len(categories)
            category_offsets = torch.tensor([0] + categories[:-1]).cumsum(0)
            self.register_buffer('category_offsets', category_offsets)
            # 범주형 변수를 768 차원으로 임베딩
            self.category_embeddings = nn.Embedding(sum(categories), d_token)
            nn_init.kaiming_uniform_(self.category_embeddings.weight, a=math.sqrt(5))

        # 수치형 변수를 768 차원으로 변환하는 가중치
        self.weight = nn.Parameter(Tensor(d_numerical, d_token))
        self.bias = nn.Parameter(Tensor(d_bias, d_token)) if bias else None
        
        # 가중치 초기화
        nn_init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            nn_init.kaiming_uniform_(self.bias, a=math.sqrt(5))

    def forward(self, x_num: Tensor, x_cat: ty.Optional[Tensor]) -> Tensor:
        x_some = x_num if x_cat is None else x_cat
        assert x_some is not None
                
        # 수치형 변수 임베딩
        x = self.weight[None] * x_num[:, :, None]  # (batch_size, n_numerical, d_token)
        
        # 범주형 변수 임베딩 추가
        if x_cat is not None:
            cat_embeddings = self.category_embeddings(x_cat + self.category_offsets[None])
            x = torch.cat([x, cat_embeddings], dim=1)  # (batch_size, n_total_features, d_token)
        
        # 바이어스 추가
        if self.bias is not None:
            x = x + self.bias[None]
            
        return x
    

class Image_Tabular_Transformer(nn.Module):
    def __init__(self, embed_dim = 768, # DINO encoder output dim
                 d_numerical = 3, # age, kps, who_grade
                 categories = [2, 2, 2, 2, 2], # sex, gbl, idh, mgmt, eor
                 tt_bias = True,
                 depth=6, num_heads=8, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop_rate=0., attn_drop_rate=0., 
                 norm_layer=nn.LayerNorm,
                 **kwargs):
        super().__init__()

        
        self.blocks = nn.ModuleList([ 
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        self.d_numerical = d_numerical
        self.categories = categories
        self.tabular_tokenizer = TabularTokenizer(d_numerical=self.d_numerical, categories=self.categories, d_token=embed_dim, bias=tt_bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def prepare_tokens(self, x):
        # [expected] Used for transforming the visual feature to a meaningfull feature.
        # change to batch
        B, len, dim = x.shape
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        return x
    
    def forward(self, tabular, image_feat):
        x_stages = []
        # Dimension change & Token added
        # tabular feature 추출
        tabular_feat = self.tabular_tokenizer(tabular[:, :self.d_numerical], tabular[:, self.d_numerical:].long()) # tabular 순서 numerical, category 순으로 정렬 필요
        image_feat=image_feat.unsqueeze(1) # (B, 1, dim)
        x = torch.cat((image_feat, tabular_feat), dim=1) # image, tabular 토큰들 concat (B, tabular_count + 1, dim)
        # image feature 추출  
        x = self.prepare_tokens(x) # cls token 추가 (B, tabular_count + 2, dim)
        for idx_stage, blk in enumerate(self.blocks): # 멀티헤드 어텐션
            x = blk(x) 
            x_stages.append(x)
            # print(f"\n{x.grad_fn}")
            
        x_last_stage = self.norm(x)
        output = x_last_stage[:,0,:] # cls토큰의 임베딩
        return output 
    

    def get_last_selfattention(self, x): # attention weight 확인
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output
    
class Image_Tabular_Transformer_HazardNet(nn.Module):
  def __init__(self, args, base_model, d_numerical, categories):
    super().__init__()
    self.base_model = base_model
    layer_names = []
    for name, module in base_model.named_modules():
        layer_names.append(name)

    # self.output_dim = args.n_intervals # 19
    self.args=args
    self.num_Event = 1
      
    # if args.net_architect == 'DenseNet':
    #    layers.append(self.model['class_layers'][:2])

    if args.net_architect == 'SEResNext50':
      self.base_model._modules[layer_names[-1]] = nn.Identity()
      image_num_out = 2048
    elif args.net_architect == 'DenseNet':
      self.base_model.class_layers.flatten= nn.Identity()
      self.base_model.class_layers.out= nn.Identity()
      image_num_out = 1024
    elif args.net_architect == 'resnet50_cbam':
      self.base_model._modules[layer_names[-1]] = nn.Identity()
      image_num_out = 2048
    elif args.net_architect == 'VisionTransformer':
      self.base_model._modules[layer_names[-1]] = nn.Identity()
      image_num_out = 1024
    elif args.net_architect == 'pretrained_resnet':
      self.base_model._modules[layer_names[-1]] = nn.Identity()
      image_num_out = 2048
    elif args.net_architect == 'DINO':
      image_num_out = 768

    num_out = image_num_out
    self.transformer_encoder = Image_Tabular_Transformer(embed_dim=num_out, d_numerical=d_numerical, categories=categories, depth=self.args.tf_depth)
    # self.layer1 = nn.ModuleList(layers)
    # self.flatten = nn.Flatten()
    # self.prophazards = PropHazards(self.num_out, self.args.num_category) # (size_in = args.last_size[args.net_architect], size_out = self.output_dim)
    # self.hazards = Hazards(self.num_out, self.args.num_category)
    self.hazards_net=HazardNetwork(self.args, num_out + 1, self.args.num_category) # len([1,3,6,12,24,36,48,60]) =8

    # self.o_layer = nn.Sequential(
    #         nn.Linear(self.num_out, self.args.num_category),
    #         nn.Softmax(dim=1)
    #     )
    
  def forward(self, tabular, image):
    # for layer in self.layer1:
    #    x = layer(x)
    # print(f'x.size:{x.size()}')
    # # pdb.set_trace()



    # x= self.base_model(x)

    # # x = self.flatten(x)

    # # x = self.prophazards(x)
    # x=self.hazards(x)

    
    # # for risk function
    # # x = x.reshape(-1, self.num_Event * self.num_out)
    # # OUTPUT layers
    # # x = self.o_layer(x)
    # x = x.reshape(-1, self.args.num_category)
    
    # pdb.set_trace()
    batch_size= tabular.shape[0]
    # x=torch.unsqueeze(x,1).expand(x.shape[0],self.args.num_category, x.shape[1], x.shape[2], x.shape[3],x.shape[4]).reshape(-1, x.shape[1], x.shape[2], x.shape[3],x.shape[4]).to(self.args.device)
    # (B,C,W,H) -> (B,1,C,W,H) -> (B,num_category,C,W,H) -> (B*num_category,C,W,H)
    image_feat = self.base_model(image)

    combined_feat = self.transformer_encoder(tabular, image_feat)
    # x.shape: (B*num_category, num_out)

    # x = self.flatten(x)
    combined_feat=torch.unsqueeze(combined_feat,1).expand(combined_feat.shape[0],self.args.num_category, combined_feat.shape[1]).reshape(-1, combined_feat.shape[1]).to(self.args.device)
    # x = self.prophazards(x)
    combined_feat=self.hazards_net(combined_feat,batch_size)

    combined_feat=combined_feat.reshape(batch_size, self.args.num_category)
    # for risk function
    # x = x.reshape(-1, self.num_Event * self.num_out)
    # OUTPUT layers
    # x = self.o_layer(x)
    # x = x.reshape(-1, self.args.num_category)


    return combined_feat