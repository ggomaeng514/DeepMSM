# for DINO
import argparse
import logging
import os
import sys
import numpy as np

from omegaconf import OmegaConf

import torch

import vision_transformer_3d as vits_3d

def build_model_3d(config):
    vit_kwargs = dict(
        img_size=config.img_size,
        patch_size=config.patch_size,
        in_chans=config.in_chans,
        ffn_layer=config.ffn_layer,
        block_chunks=config.block_chunks,
        qkv_bias=config.qkv_bias,
        proj_bias=config.proj_bias,
        ffn_bias=config.ffn_bias,
        num_register_tokens=config.num_register_tokens,
        interpolate_offset=config.interpolate_offset,
        interpolate_antialias=config.interpolate_antialias,
    )
    teacher = vits_3d.__dict__[config.arch](**vit_kwargs)
    return teacher

def load_pretrained_weights(model, pretrained_weights, checkpoint_key):
    state_dict = torch.load(pretrained_weights, map_location="cpu")
    print(state_dict.keys())
    if checkpoint_key is not None and checkpoint_key in state_dict:
        state_dict = state_dict[checkpoint_key]
    # remove `module.` prefix
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    return model


def build_model_for_eval_3d(config, pretrained_weights):
    model = build_model_3d(config.Model)
    model = load_pretrained_weights(model, pretrained_weights, "teacher")
    model.eval()
    model.cuda()
    return model


def setup_and_build_model_3d(args, config):
    model = build_model_for_eval_3d(config, args.pretrained_weights)
    autocast_dtype = torch.half
    return model, autocast_dtype

def get_args_parser():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    return parser

def get_args_parser_in_notebook():
    parser = get_args_parser()
    
    # Manually provide arguments in the notebook
    args = parser.parse_args([
        '--config-file', './configs/vit_3d.yaml',
        '--pretrained-weights', './checkpoints/UPennTCGA_data_checkpoint.pth',
    ])
    
    return args
