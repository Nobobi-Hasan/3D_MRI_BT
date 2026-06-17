# src/dataset.py

import json
import os
from monai.data import Dataset, CacheDataset, DataLoader
import src.config as config

def load_split_records(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_brats_dataset(records, transforms, use_cache=True):
    if use_cache and config.CACHE_RATE > 0:
        return CacheDataset(
            data=records,
            transform=transforms,
            cache_rate=config.CACHE_RATE,
            num_workers=config.NUM_WORKERS
        )
    return Dataset(data=records, transform=transforms)

def create_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True   # pinned memory for rapid GPU transfer
    )