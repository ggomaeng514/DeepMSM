# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from glob import glob
from typing import Callable, List, Optional, Tuple, Union
import json
import random
import pandas as pd
import torch
import nibabel as nib

import numpy as np
from torch.utils.data import Dataset


logger = logging.getLogger("dinov2")
_Target = int


class MRI(Dataset):
    def __init__(self, dataroot, phase, transform):
        super().__init__()
        self.dataroot = dataroot
        self.phase = phase
        self.transform = transform

        self.dataset = glob(os.path.join(self.dataroot, '*'))
        
    def __getitem__(self, index):
        data_path = self.dataset[index]
        t1, t1ce, t2, flair, mask, tumor_mask, patient_id = self._single_scan(data_path)
        data = self.transform({'t1': t1, 't1ce': t1ce, 't2': t2, 'flair': flair, 'mask': mask, 'tumor_mask': tumor_mask, 'patient_id': patient_id})
        
        return data
    
    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.dataset)
        
    def _single_scan(self, patient_dir):
        patient_id = patient_dir.split('/')[-2]
        t1_path = os.path.join(patient_dir, 't1.nii.gz')
        t1ce_path = os.path.join(patient_dir, 't1ce.nii.gz')
        t2_path = os.path.join(patient_dir, 't2.nii.gz')
        flair_path = os.path.join(patient_dir, 'flair.nii.gz')
        mask_path = os.path.join(patient_dir, 'brain_mask.nii.gz')
        tumor_mask_path = os.path.join(patient_dir, 'tumor_mask.nii.gz')
        
        t1 = nib.load(t1_path).get_fdata()
        t1ce = nib.load(t1ce_path).get_fdata()
        t2 = nib.load(t2_path).get_fdata()
        flair = nib.load(flair_path).get_fdata()
        mask = nib.load(mask_path).get_fdata()
        tumor_mask = nib.load(tumor_mask_path).get_fdata()
        
        return [t1, t1ce, t2, flair, mask, tumor_mask, patient_id]
