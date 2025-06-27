#import cv2
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as TF
import PIL.Image as Image
import torch
from torch import nn
import os
import pdb
# from utils import get_n_intervals

import numpy as np

from monai.transforms import RandFlip, Rand3DElastic, RandAffine, RandGaussianNoise, AdjustContrast, RandSpatialCrop # Rand3DElastic
import torchio as tio
from sklearn.preprocessing import StandardScaler



class Eval_dataset(nn.Module):
  def __init__(self, df, main_args, dataset_name=None): # ['27179925', '45163562', 'UPENN-GBM-00291_11', '42488471', 'UPENN-GBM-00410_11', '28802482']
    self.dataset_name = dataset_name
    self.df = df
    self.df['id'] = self.df['id'].astype(str)
    self.args = main_args
    # print(self.df.shape) # (890, 39) # (223, 39)
    # self.data_dir=args.data_dir

    self.time  = np.asarray(self.df[['duration_'+self.args.spec_event]])
    self.label = np.asarray(self.df[['event_'+self.args.spec_event]])

    # self.ID = np.array(df.index.values, dtype=str)

  
  def concat_seq_img(self, x):
      return torch.cat([x[sequence][tio.DATA] for sequence in self.args.sequence], axis=0)
  
  def __len__(self):
    return len(self.df) # 결국 df 로 index 를 하기 때문에 dataset의 길이도 len(df): df를 train_df, val_df 넣는것에 따라 dataset이 train_set, val_set이 됨.


  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    # ID = self.df.iloc[idx].ID #name 맞는지 확인 필요
    # kfold = self.df['kfold'][idx]

    t=self.time[idx]
    k=self.label[idx]


    return t, k

  
  
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


def f_get_fc_mask2(time, label, num_Category):
    time = np.asarray(time)
    label = np.asarray(label)

    mask = np.zeros([np.shape(time)[0], num_Category])
    for i in range(np.shape(time)[0]):
        if label[i,0] != 0:  #not censored
            mask[i, int(time[i,0])] = 1
        else: #label[i,2]==0: censored
            mask[i, int(time[i,0]+1):] =  1 #fill 1 until from the censoring time
    return mask

def f_get_fc_mask3(time, label, num_Category):
    time = np.asarray(time)
    label = np.asarray(label)

    mask = np.zeros([np.shape(time)[0], num_Category])
    for i in range(time.shape[0]):
        time_idx = int(time[i])
        if label[i] != 0:                   # uncensor
            if time_idx != 0:               # first time pass (do nothing)
                mask[i, :time_idx] = 1      # before event time = 1
        else:                               # censor
            mask[i, :time_idx+1] = 1        # until censor time = 1
    return mask

def f_get_fc_mask4(time, label, num_Category):
    time = np.asarray(time)
    label = np.asarray(label)

    mask = np.zeros([np.shape(time)[0], num_Category])
    for i in range(np.shape(time)[0]):
        mask[i, int(time[i,0])] = 1
    return mask

class Image_N_TabularDataset(nn.Module):
  def __init__(self, df, args, scaler, t_token=False):
    # self.dataset_name = dataset_name
    self.df = df
    self.args = args
    self.data_dir=args.data_dir
    self.scaler = scaler
    self.npy_dir ='./data/jhlee'
    self.time  = np.asarray(self.df[['duration_'+self.args.spec_event]])
    self.label = np.asarray(self.df[['event_'+self.args.spec_event]])


    self.ID = np.array(df.index.values, dtype=str)
    self.feature=self.df[self.args.data_key_list]
    # print(self.feature)
    if 'EOR_str' in self.args.data_key_list:
      self.feature=pd.concat([self.feature, pd.get_dummies(self.feature['EOR_str'], prefix='EOR').astype(int)], axis=1)
      self.feature=self.feature.drop(columns=['EOR_str'])
    if 'glioma_type' in self.args.data_key_list:
      self.feature[['glioma_astrocytoma', 'glioma_gliosarcoma', 'glioma_mutant', 'glioma_oligodendro', 'glioma_wild']] = 0
      self.feature.loc[self.feature['glioma_type'] == 'astrocytoma', 'glioma_astrocytoma'] = 1
      self.feature.loc[self.feature['glioma_type'] == 'mutant', 'glioma_mutant'] = 1
      self.feature.loc[self.feature['glioma_type'] == 'oligodendro', 'glioma_oligodendro'] = 1
      self.feature.loc[self.feature['glioma_type'] == 'wild', 'glioma_wild'] = 1
      self.feature=self.feature.drop(columns=['glioma_type'])

    if t_token and 'sex' in self.args.data_key_list:
      self.feature['sex'] = pd.factorize(self.feature['sex'])[0] # 2: female, 1: male -> 1: female, 0: male

    if self.scaler != None:
      self.feature[['age']] = self.scaler.transform(self.feature[['age']])

    if 'kps' in self.args.data_key_list:
        self.feature[['kps']] = self.feature[['kps']]/ 100.0

    if not t_token:
      self.feature=self.feature.sort_index(axis=1)
    print(self.feature.keys())
    self.feature=np.asarray(self.feature)

    self.mask_1 = f_get_fc_mask2(self.time, self.label, self.args.num_category)
    self.mask_2 = f_get_fc_mask3(self.time, self.label, self.args.num_category)
    self.mask_3 = f_get_fc_mask4(self.time, self.label, self.args.num_category)

  
  def __len__(self):
    return len(self.df)


  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")


    ID = self.df.iloc[idx].id 

    subj_img_dir = os.path.join(self.data_dir, self.source_data.iloc[idx] ,str(ID))
    subdirs = [d for d in os.listdir(subj_img_dir) if os.path.isdir(os.path.join(subj_img_dir, d))]
    if not subdirs:
        raise RuntimeError(f"No subdirectories found under {subj_img_dir}")

    first_subdir = os.path.join(subj_img_dir, subdirs[0])

    # 이미지 로딩
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(first_subdir, 't1.nii.gz')),
        t2=tio.ScalarImage(os.path.join(first_subdir, 't2.nii.gz')),
        t1ce=tio.ScalarImage(os.path.join(first_subdir, 't1ce.nii.gz')),
        flair=tio.ScalarImage(os.path.join(first_subdir, 'flair.nii.gz')),
        brain=tio.ScalarImage(os.path.join(first_subdir, 'brain_mask.nii.gz')),
        tumor=tio.ScalarImage(os.path.join(first_subdir, 'tumor_mask.nii.gz')),
        mask=tio.LabelMap(os.path.join(first_subdir, 'brain_mask.nii.gz')),
    )

    t1 = subject['t1']
    t1ce = subject['t1ce']
    t2 = subject['t2']
    flair = subject['flair']
    mask = subject['mask']
    brain = subject['brain']
    tumor = subject['tumor']
    t1_signal = t1.data[mask.data > 0]
    t1ce_signal = t1ce.data[mask.data > 0]
    t2_signal = t2.data[mask.data > 0]
    brain_signal = brain.data[mask.data > 0]
    tumor_signal = tumor.data[mask.data > 0]
    flair_signal = flair.data[mask.data > 0]
    meanv_t1, stdv_t1 = t1_signal.mean(), t1_signal.std()
    meanv_t1ce, stdv_t1ce = t1ce_signal.mean(), t1ce_signal.std()
    meanv_t2, stdv_t2 = t2_signal.mean(), t2_signal.std()
    meanv_flair, stdv_flair = flair_signal.mean(), flair_signal.std()
    meanv_brain, stdv_brain = brain_signal.mean(), brain_signal.std()
    meanv_tumor, stdv_tumor = tumor_signal.mean(), tumor_signal.std()
    t1 = (t1.data - meanv_t1) / (stdv_t1 + 1e-8)
    t1ce = (t1ce.data - meanv_t1ce) / (stdv_t1ce + 1e-8)
    t2 = (t2.data - meanv_t2) / (stdv_t2 + 1e-8)
    flair = (flair.data - meanv_flair) / (stdv_flair + 1e-8)
    brain = (brain.data - meanv_brain) / (stdv_brain + 1e-8)
    tumor = (tumor.data - meanv_tumor) / (stdv_tumor + 1e-8)
    img = torch.cat([t1, t1ce, t2, flair], dim=0)
    img = img * mask.data
    img = img.clamp(-2.5, 2.5)
    
    if self.transforms:
      subject = self.transforms(subject)
    

    t=self.time[idx]
    k=self.label[idx]
    mk1=self.mask_1[idx]
    mk2=self.mask_2[idx]
    mk3=self.mask_3[idx]

    if self.aug_transform:
          img = self.augment(img)
    img = torch.Tensor(img).to(dtype=torch.float)

    tabular_x=self.feature[idx]

    return ID, tabular_x, img, t, k, mk1, mk2, mk3

  
class Image_N_TabularDatasetFP(nn.Module):
  def __init__(self, df, args, scaler, t_token=False):
    self.df = df
    self.args = args
    self.data_dir=args.data_dir
    self.scaler = scaler
    self.npy_dir ='./data/jhlee'
    self.time  = np.asarray(self.df[['duration_'+self.args.spec_event]])
    self.label = np.asarray(self.df[['event_'+self.args.spec_event]])


    self.ID = np.array(df.index.values, dtype=str)
    self.feature=self.df[self.args.data_key_list]
    if 'EOR_str' in self.args.data_key_list:
      self.feature=pd.concat([self.feature, pd.get_dummies(self.feature['EOR_str'], prefix='EOR').astype(int)], axis=1)
      self.feature=self.feature.drop(columns=['EOR_str'])
    if 'glioma_type' in self.args.data_key_list:
      self.feature[['glioma_astrocytoma', 'glioma_gliosarcoma', 'glioma_mutant', 'glioma_oligodendro', 'glioma_wild']] = 0
      self.feature.loc[self.feature['glioma_type'] == 'astrocytoma', 'glioma_astrocytoma'] = 1
      self.feature.loc[self.feature['glioma_type'] == 'mutant', 'glioma_mutant'] = 1
      self.feature.loc[self.feature['glioma_type'] == 'oligodendro', 'glioma_oligodendro'] = 1
      self.feature.loc[self.feature['glioma_type'] == 'wild', 'glioma_wild'] = 1
      self.feature=self.feature.drop(columns=['glioma_type'])

    if t_token and 'sex' in self.args.data_key_list:
      self.feature['sex'] = pd.factorize(self.feature['sex'])[0] # 2: female, 1: male -> 1: female, 0: male

    if self.scaler != None:
      self.feature[['age']] = self.scaler.transform(self.feature[['age']])

    if 'kps' in self.args.data_key_list:
        self.feature[['kps']] = self.feature[['kps']]/ 100.0

    if not t_token:
      self.feature=self.feature.sort_index(axis=1)
    print(self.feature.keys())
    self.feature=np.asarray(self.feature)

    self.mask_1 = f_get_fc_mask2(self.time, self.label, self.args.num_category)
    self.mask_2 = f_get_fc_mask3(self.time, self.label, self.args.num_category)
    self.mask_3 = f_get_fc_mask4(self.time, self.label, self.args.num_category)

  
  def __len__(self):
    return len(self.df)


  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")

    ID = self.df.iloc[idx].id
    SOURCE_DATA = self.df.iloc[idx].source_data
    if SOURCE_DATA == 'UPenn':
      subj_npy_dir = os.path.join(self.npy_dir, self.args.cohort , "UPENN", f"{str(ID)}.npy")
    elif SOURCE_DATA == 'severance':
      subj_npy_dir = os.path.join(self.npy_dir, self.args.cohort , "Severance", f"{str(ID)}.npy")
    else:
      subj_npy_dir = os.path.join(self.npy_dir, self.args.cohort , SOURCE_DATA, f"{str(ID)}.npy")
    img = np.load(subj_npy_dir)

    t=self.time[idx]
    k=self.label[idx]
    mk1=self.mask_1[idx]
    mk2=self.mask_2[idx]
    mk3=self.mask_3[idx]

    img = torch.Tensor(img).to(dtype=torch.float)

    tabular_x=self.feature[idx]

    return ID, tabular_x, img, t, k, mk1, mk2, mk3


  def __init__(self, df, args, dataset_name, scaler, transforms=None, aug_transform=False, contrastive=False): # ['27179925', '45163562', 'UPENN-GBM-00291_11', '42488471', 'UPENN-GBM-00410_11', '28802482']
    self.dataset_name = dataset_name
    self.df = df
    self.args = args
    # self.main_args = main_args
    # print(self.df.shape) # (890, 39) # (223, 39)
    self.data_dir=args.data_dir
    self.scaler = scaler
    self.img_dir=os.path.join(args.data_dir, self.dataset_name,"VIT", f'{args.compart_name}_BraTS_v2') if self.args.net_architect == 'VisionTransformer' else os.path.join(args.data_dir, self.dataset_name, f'{args.compart_name}_BraTS_v2') # 'SNUH_UPenn_TCGA_severance'
    # self.img_dir=os.path.join(args.data_dir, self.dataset_name,"VIT", f'{args.compart_name}_BraTS') if self.args.net_architect == 'VisionTransformer' else os.path.join(args.data_dir, self.dataset_name, f'{args.compart_name}_BraTS') # 'SNUH_UPenn_TCGA_severance'

    self.time  = np.asarray(self.df[['duration_'+self.args.spec_event]])
    self.label = np.asarray(self.df[['event_'+self.args.spec_event]])

    # pdb.set_trace()
    # self.source_data = self.df['source_data']
    self.source_data = self.df['source_data'].str.replace('severance', 'Severance')
    self.ID = np.array(df.index.values, dtype=str)

    # ✅ tabular 전처리 직접 삽입
    self.feature = self.df[self.args.data_key_list].copy()

    if 'EOR_str' in self.args.data_key_list:
        self.feature = pd.concat(
            [self.feature, pd.get_dummies(self.feature['EOR_str'], prefix='EOR').astype(int)],
            axis=1
        )
        self.feature = self.feature.drop(columns=['EOR_str'])

    if 'glioma_type' in self.args.data_key_list:
        self.feature[['glioma_astrocytoma', 'glioma_gliosarcoma', 'glioma_mutant', 'glioma_oligodendro', 'glioma_wild']] = 0
        self.feature.loc[self.feature['glioma_type'] == 'astrocytoma', 'glioma_astrocytoma'] = 1
        self.feature.loc[self.feature['glioma_type'] == 'mutant', 'glioma_mutant'] = 1
        self.feature.loc[self.feature['glioma_type'] == 'oligodendro', 'glioma_oligodendro'] = 1
        self.feature.loc[self.feature['glioma_type'] == 'wild', 'glioma_wild'] = 1
        self.feature = self.feature.drop(columns=['glioma_type'])

    # ✳️ 성별 인코딩은 사용하지 않으므로 아래 줄 생략해도 됨
    # self.feature['sex'] = pd.factorize(self.feature['sex'])[0]

    # ✳️ scaler 적용: age에만 적용 (main_args에서 가져옴)
    if self.scaler is not None and 'age' in self.feature.columns:
        self.feature[['age']] = self.scaler.transform(self.feature[['age']])

    # ✳️ kps 정규화
    if 'kps' in self.args.data_key_list:
        self.feature[['kps']] = self.feature[['kps']] / 100.0

    # ✳️ 컬럼 정렬 및 numpy 변환
    self.feature = self.feature.sort_index(axis=1)
    print(self.feature.keys())
    self.feature = np.asarray(self.feature)


    ###################################


    # num_Category = int(np.max(self.time) * 1.2)        #to have enough time-horizon
    # num_Event = int(len(np.unique(self.label)) - 1) #only count the number of events (do not count censoring as an event)

    self.mask_1 = f_get_fc_mask2(self.time, self.label, self.args.num_category)
    self.mask_2 = f_get_fc_mask3(self.time, self.label, self.args.num_category)
    self.mask_3 = f_get_fc_mask4(self.time, self.label, self.args.num_category)


    self.transforms = transforms
    self.aug_transform = aug_transform
    self.contrastive= contrastive #추가

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
    if self.contrastive: #추가
        img1 = self.flipper1(img)
        img1 = self.flipper2(img1)
        # img1 = self.flipper3(img1)
        img1 = self.rand_elastic(img1)
        img1 = self.contrast(img1)
        img1 = self.gaussian(img1)

        img2 = self.flipper1(img)
        img2 = self.flipper2(img2)
        # img2 = self.flipper3(img2)
        img2 = self.rand_elastic(img2)
        img2 = self.contrast(img2)
        img2 = self.gaussian(img2)

        return [img1,img2]
    else:
        img1 = self.flipper1(img)
        img1 = self.flipper2(img1)
        # img1 = self.flipper3(img1)
        img1 = self.rand_elastic(img1)
        img1 = self.contrast(img1)
        img1 = self.gaussian(img1)
    
        return img1


  def __getitem__(self, idx):
    if type(idx) is not int:
      raise ValueError(f"Need `index` to be `int`. Got {type(idx)}.")



    ID = self.df.iloc[idx].id #name 맞는지 확인 필요
    # ID = self.df.iloc[idx].ID #name 맞는지 확인 필요
    # kfold = self.df['kfold'][idx]
    # /mnt/hdd1/chlee/data/jhlee/data/Severance
    # subj_img_dir = os.path.join(self.img_dir, str(ID))
    subj_img_dir = os.path.join(self.data_dir, self.source_data.iloc[idx] ,str(ID))
    # self.source_data
    # subj_img_dir = os.path.join(self.data_dir, self.dataset_name,f'{self.compart_name}_BraTS',str(ID)) #수정
    # /mnt/hdd1/chlee/data/jhlee/data/Severance/Sev001/20171205/tumor_mask.nii.gz
            
    # subject = tio.Subject(
    #     t1=tio.ScalarImage(os.path.join(subj_img_dir, f't1_{self.compart_name}.nii.gz')), # t1_seg.nii.gz
    #     t2=tio.ScalarImage(os.path.join(subj_img_dir, f't2_{self.compart_name}.nii.gz')), 
    #     t1ce=tio.ScalarImage(os.path.join(subj_img_dir, f't1ce_{self.compart_name}.nii.gz')), 
    #     flair=tio.ScalarImage(os.path.join(subj_img_dir, f'flair_{self.compart_name}.nii.gz')), 
    #     )   
    # id 밑에 있는 첫 번째 하위 디렉토리 찾기 (예: '05')
    subdirs = [d for d in os.listdir(subj_img_dir) if os.path.isdir(os.path.join(subj_img_dir, d))]
    if not subdirs:
        raise RuntimeError(f"No subdirectories found under {subj_img_dir}")

    # 하위 디렉토리 중 첫 번째 꺼 쓴다. 혹은 for loop 돌려도 됨.
    first_subdir = os.path.join(subj_img_dir, subdirs[0])

    # 이미지 로딩
    subject = tio.Subject(
        t1=tio.ScalarImage(os.path.join(first_subdir, 't1.nii.gz')),
        t2=tio.ScalarImage(os.path.join(first_subdir, 't2.nii.gz')),
        t1ce=tio.ScalarImage(os.path.join(first_subdir, 't1ce.nii.gz')),
        flair=tio.ScalarImage(os.path.join(first_subdir, 'flair.nii.gz')),
        brain=tio.ScalarImage(os.path.join(first_subdir, 'brain_mask.nii.gz')),
        tumor=tio.ScalarImage(os.path.join(first_subdir, 'tumor_mask.nii.gz')),
        mask=tio.LabelMap(os.path.join(first_subdir, 'brain_mask.nii.gz')),
    )

    
    # pdb.set_trace()
    t1 = subject['t1']
    t1ce = subject['t1ce']
    t2 = subject['t2']
    flair = subject['flair']
    mask = subject['mask']
    brain = subject['brain']
    tumor = subject['tumor']
    t1_signal = t1.data[mask.data > 0]
    t1ce_signal = t1ce.data[mask.data > 0]
    t2_signal = t2.data[mask.data > 0]
    brain_signal = brain.data[mask.data > 0]
    tumor_signal = tumor.data[mask.data > 0]
    flair_signal = flair.data[mask.data > 0]
    meanv_t1, stdv_t1 = t1_signal.mean(), t1_signal.std()
    meanv_t1ce, stdv_t1ce = t1ce_signal.mean(), t1ce_signal.std()
    meanv_t2, stdv_t2 = t2_signal.mean(), t2_signal.std()
    meanv_flair, stdv_flair = flair_signal.mean(), flair_signal.std()
    meanv_brain, stdv_brain = brain_signal.mean(), brain_signal.std()
    meanv_tumor, stdv_tumor = tumor_signal.mean(), tumor_signal.std()
    t1 = (t1.data - meanv_t1) / (stdv_t1 + 1e-8)
    t1ce = (t1ce.data - meanv_t1ce) / (stdv_t1ce + 1e-8)
    t2 = (t2.data - meanv_t2) / (stdv_t2 + 1e-8)
    flair = (flair.data - meanv_flair) / (stdv_flair + 1e-8)
    brain = (brain.data - meanv_brain) / (stdv_brain + 1e-8)
    tumor = (tumor.data - meanv_tumor) / (stdv_tumor + 1e-8)
    img = torch.cat([t1, t1ce, t2, flair], dim=0)
    img = img * mask.data
    img = img.clamp(-2.5, 2.5)
    
    if self.transforms:
      subject = self.transforms(subject)
    
    # img = self.concat_seq_img(subject)
    # img=subject
    

    t=self.time[idx]
    k=self.label[idx]
    mk1=self.mask_1[idx]
    mk2=self.mask_2[idx]
    mk3=self.mask_3[idx]


    if self.aug_transform:
      img = self.augment(img) #contrastive == True면 img는 리스트로 같은 이미지의 다른 transform을 적용한 이미지 두개가 묶여서 나옴

    img = torch.Tensor(img.data).to(dtype=torch.float)

    tabular_x=self.feature[idx]

    return ID, tabular_x, img, t, k, mk1, mk2, mk3, brain, tumor