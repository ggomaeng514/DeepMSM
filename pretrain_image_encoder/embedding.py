# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

from mri_dinov2.data import make_data_loader, make_dataset_mri
from mri_dinov2.data.transforms_3d import make_classification_eval_transform
from mri_dinov2.eval.utils import ModelWithNormalize
from mri_dinov2.eval.setup import setup_and_build_model_3d
from mri_dinov2.utils.config import get_cfg_from_args

logger = logging.getLogger("dinov2")


def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    # set batch_size and num_workers to 1 for embedding
    parser.add_argument("--batch-size", default=16, type=int, help="Per-GPU batch-size")
    parser.add_argument("--num-workers", default=8, type=int, help="Number of workers")

    return parser


@torch.inference_mode()
def extract_features_with_dataloader(model, data_loader, sample_count, gather_on_cpu=False):
    gather_device = torch.device("cpu") if gather_on_cpu else torch.device("cuda")
    # metric_logger = MetricLogger(delimiter="  ")
    features = {}
    for data in tqdm(data_loader):
        samples = data["transformed_image"].cuda(non_blocking=True)
        features_rank = model(samples).float()
        
        patient_id = data["patient_id"]
        
        for i, p_i in enumerate(patient_id):
            key = f"{p_i}"
            features[key] = features_rank[i].cpu().detach().numpy()
        
    logger.info(f"Number of Features: {len(features)}")
    return features
    

def do_embedding(args, cfg, model):
    dataset = make_dataset_mri(
        dataset_path=cfg.train.dataset_path,
        transform=make_classification_eval_transform(cfg.crops.global_crops_size),
    )
    sample_count = len(dataset)
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=False,
        shuffle=False,
        persistent_workers=True,
    )
    
    model = ModelWithNormalize(model)
    
    logger.info("Extracting features for train set...")
    features = extract_features_with_dataloader(model, data_loader, sample_count)
    
    for key, value in features.items():
        patient_id = key
        save_path = os.path.join(args.output_dir, f"{patient_id}.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, value)


def main(args):
    cfg = get_cfg_from_args(args)
    
    model, autocast_dtype = setup_and_build_model_3d(args)

    with torch.cuda.amp.autocast(dtype=autocast_dtype):
        do_embedding(args, cfg, model)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
