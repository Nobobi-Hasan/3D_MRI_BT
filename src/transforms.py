# src/transforms.py

import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    MapLabelValued,
    CastToTyped,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandAdjustGammad
)
import src.config as config

def get_train_transforms():
    return Compose([
        # Loading NIfTI volumes from paths
        LoadImaged(keys=["image", "label"]),
        # Enforcing channel-first tensor format
        EnsureChannelFirstd(keys=["image", "label"]),
        # Normalizing intensity per channel using non-zero voxel stats
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # Mapping original label 4 to contiguous label 3
        MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
        
        # Balanced sampling: Ensures the network sees tumor regions frequently
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=config.PATCH_SIZE,
            pos=1,
            neg=1,
            num_samples=1,
            image_key="image",
            image_threshold=0,
        ),

        # Spatial Augmentations
        RandAffined(
            keys=["image", "label"],
            mode=("bilinear", "nearest"),
            prob=0.2,
            spatial_size=config.PATCH_SIZE,
            rotate_range=(np.pi/12, np.pi/12, np.pi/12),
            scale_range=(0.1, 0.1, 0.1)
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

        # Intensity Augmentations (Applied ONLY to image)
        RandGaussianNoised(keys=["image"], prob=0.1, mean=0.0, std=0.05),
        RandGaussianSmoothd(
            keys=["image"], 
            prob=0.1, 
            sigma_x=(0.5, 1.0),
            sigma_y=(0.5, 1.0),
            sigma_z=(0.5, 1.0)
        ),
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.15),
        RandAdjustGammad(keys=["image"], gamma=(0.7, 1.5), prob=0.15),

        # Casting data to precise torch types
        CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.int64])
    ])

def get_val_transforms():
    return Compose([
        # Loading NIfTI volumes from paths
        LoadImaged(keys=["image", "label"]),
        # Enforcing channel-first tensor format
        EnsureChannelFirstd(keys=["image", "label"]),
        # Normalizing intensity per channel using non-zero voxel stats
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        # Mapping original label 4 to contiguous label 3
        MapLabelValued(keys="label", orig_labels=[4], target_labels=[3]),
        # Casting data to precise torch types
        CastToTyped(keys=["image", "label"], dtype=[torch.float32, torch.int64])
    ])