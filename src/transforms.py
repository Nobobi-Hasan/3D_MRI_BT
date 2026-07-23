# src/transforms.py

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    MapLabelValued,
    CastToTyped
)
from monai.transforms import MapTransform
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
        # Randomly cropping 3D sub-volume patches to fit memory constraints
        RandSpatialCropd(keys=["image", "label"], roi_size=config.PATCH_SIZE, random_size=False),
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