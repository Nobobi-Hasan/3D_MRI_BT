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

class RandModalityDropoutd(MapTransform):
    """Custom dictionary transform for independent modality channel dropout."""
    def __init__(self, keys, prob=0.2):
        super().__init__(keys)
        self.prob = prob

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            num_channels = img.shape[0]
            
            # Executing threshold check for channel dropout
            if isinstance(img, torch.Tensor):
                drop_mask = torch.rand(num_channels, device=img.device) < self.prob
                if drop_mask.all():
                    keep_idx = torch.randint(0, num_channels, (1,), device=img.device).item()
                    drop_mask[keep_idx] = False
                
                img = img.clone()
                img[drop_mask] = 0.0
            else:
                import numpy as np
                drop_mask = np.random.rand(num_channels) < self.prob
                if drop_mask.all():
                    keep_idx = np.random.randint(0, num_channels)
                    drop_mask[keep_idx] = False
                
                img = img.copy()
                img[drop_mask] = 0.0
                
            d[key] = img
        return d

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
        # Simulating missing sequence modalities via randomized dropout
        RandModalityDropoutd(keys="image", prob=config.MODALITY_DROPOUT_PROB),
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