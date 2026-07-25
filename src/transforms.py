# src/transforms.py

import torch
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    RandSpatialCropd,
    MapLabelValued,
    CastToTyped,
    RandFlipd,
    RandRotated,
    RandScaleIntensityd,
    RandShiftIntensityd
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

        # Add Spatial Augmentations
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),

        # +/- ~10 degree rotations (0.175 radians)
        RandRotated(
            keys=["image", "label"], 
            range_x=0.175, range_y=0.175, range_z=0.175, 
            prob=0.3, 
            keep_size=True, 
            mode=("bilinear", "nearest") # Bilinear for MRI, Nearest for integer labels
        ),

        # Add Intensity Augmentations (Applied ONLY to image, NEVER the label)
        RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
        RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),

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