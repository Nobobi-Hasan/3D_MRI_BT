# src/dataset.py

import json
import os
# from monai.data import Dataset, CacheDataset, DataLoader
from monai.data import Dataset, PersistentDataset, DataLoader
import src.config as config
from src.transforms import get_train_transforms, get_val_transforms

def load_split_records(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def create_brats_dataset(records, transforms, use_cache=False):
    """Returns a standard clean dataset to bypass disk-write performance chokeholds."""
    return Dataset(data=records, transform=transforms)

def create_dataloader(dataset, batch_size, shuffle=False):
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True   # pinned memory for rapid GPU transfer
    )

def get_brats_dataloaders():
    """Assembles and returns the final train and validation dataloaders."""
    # Load patient record dictionary splits
    splits = load_split_records(config.DATA_SPLIT_JSON)
    train_records = splits["train"]
    val_records = splits["val"]
    
    # Retrieve preconfigured MONAI transform pipelines
    train_transforms = get_train_transforms()
    val_transforms = get_val_transforms()
    
    # Construct datasets
    train_ds = create_brats_dataset(train_records, train_transforms, use_cache=True)
    val_ds = create_brats_dataset(val_records, val_transforms, use_cache=False)
    
    # Wrap in MONAI data loaders
    train_loader = create_dataloader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader
    
def get_test_dataloader():
    """Assembles and returns the final test dataloader."""
    # Load patient record dictionary splits
    splits = load_split_records(config.DATA_SPLIT_JSON)
    test_records = splits["test"]
    
    # Retrieve preconfigured MONAI transform pipelines (testing uses validation transforms)
    test_transforms = get_val_transforms()
    
    # Construct dataset
    test_ds = create_brats_dataset(test_records, test_transforms, use_cache=False)
    
    # Wrap in MONAI data loader
    test_loader = create_dataloader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    
    return test_loader