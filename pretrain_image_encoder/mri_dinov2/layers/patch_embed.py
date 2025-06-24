# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# References:
#   https://github.com/facebookresearch/dino/blob/master/vision_transformer.py
#   https://github.com/rwightman/pytorch-image-models/tree/master/timm/layers/patch_embed.py

from typing import Callable, Optional, Tuple, Union

from torch import Tensor
import torch.nn as nn


# def make_2tuple(x):
#     if isinstance(x, list):
#         assert len(x) == 2
#         return x

#     if isinstance(x, tuple):
#         assert len(x) == 2
#         return x

#     assert isinstance(x, int)
#     return (x, x)

def make_2tuple(x):
    if isinstance(x, int):
        return (x, x)
    else:
        return tuple(x)

def make_3tuple(x):
    if isinstance(x, int):
        return (x, x, x)
    else:
        return tuple(x)


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)

    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        _, _, H, W = x.shape
        patch_H, patch_W = self.patch_size

        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"

        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class PatchEmbed3D(nn.Module):
    """
    3D image to patch embedding: (B,C,W,H,D) -> (B,N,D)
    
    Args:
        img_size: Image size.
        patch_size: Patch token size.
        in_chans: Number of input image channels.
        embed_dim: Number of linear projection output channels.
        norm_layer: Normalization layer.
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()
        
        image_WHD = make_3tuple(img_size)
        patch_WHD = make_3tuple(patch_size)
        patch_grid_size = (
            image_WHD[0] // patch_WHD[0],
            image_WHD[1] // patch_WHD[1],
            image_WHD[2] // patch_WHD[2],
        )
        
        self.img_size = image_WHD
        self.patch_size = patch_WHD
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1] * patch_grid_size[2]
        
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        
        self.flatten_embedding = flatten_embedding
        
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_WHD, stride=patch_WHD)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x: Tensor) -> Tensor:
        _, _, W, H, D = x.shape
        patch_W, patch_H, patch_D = self.patch_size
        
        assert W % patch_W == 0, f"Input image width {W} is not a multiple of patch width: {patch_W}"
        assert H % patch_H == 0, f"Input image height {H} is not a multiple of patch height {patch_H}"
        assert D % patch_D == 0, f"Input image depth {D} is not a multiple of patch depth: {patch_D}"
        
        x = self.proj(x)
        W, H, D = x.size(2), x.size(3), x.size(4)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, W, H, D, self.embed_dim)
        return x
    
    def flops(self) -> float:
        Wo, Ho, Do = self.patches_resolution
        flops = Wo * Ho * Do * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1] * self.patch_size[2])
        if self.norm is not None:
            flops += Wo * Ho * Do * self.embed_dim
        return flops
    