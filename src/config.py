# src/config.py 

import os

# =========================================================================
# 1. Directory and File Path Configurations
# =========================================================================
# Drive project root path
DRIVE_PROJECT_ROOT = "/content/drive/MyDrive/ML_Projects/3D_MRI_Brain_Tumor"

# Dataset zip file source path
ZIP_SOURCE_PATH_RAW = "/content/drive/MyDrive/ML-Datasets/BraTS2020_TrainingData.zip"
ZIP_SOURCE_PATH = "/content/drive/MyDrive/ML-Datasets/BraTS2020_TrainingData_ABC.zip"

# Local NVMe scratch space for fast unzipping
LOCAL_EXTRACT_DIR = "/content"
DATA_ROOT_DIR_RAW = os.path.join(LOCAL_EXTRACT_DIR, "MICCAI_BraTS2020_TrainingData")
DATA_ROOT_DIR = os.path.join(LOCAL_EXTRACT_DIR, "MICCAI_BraTS2020_TrainingData_ABC")

# Metadata and label files inside unzipped dataset
NAME_MAPPING_CSV = os.path.join(DATA_ROOT_DIR, "name_mapping.csv")
SURVIVAL_INFO_CSV = os.path.join(DATA_ROOT_DIR, "survival_info.csv")

# Path to save dataset split configuration
DATA_SPLIT_JSON = os.path.join(DRIVE_PROJECT_ROOT, "data/data_splits.json")

# Saved checkpoints directory for session resumes
CHECKPOINT_DIR = os.path.join(DRIVE_PROJECT_ROOT, "checkpoints")
LATEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "latest_model.pth")
BEST_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_model.pth")

# Expert pre-training checkpoints directory and paths
EXPERT_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, "experts")
PRETRAINED_EXPERT_PATHS = {
    "t1": os.path.join(EXPERT_CHECKPOINT_DIR, "best_expert_t1.pth"),
    "t1ce": os.path.join(EXPERT_CHECKPOINT_DIR, "best_expert_t1ce.pth"),
    "t2": os.path.join(EXPERT_CHECKPOINT_DIR, "best_expert_t2.pth"),
    "flair": os.path.join(EXPERT_CHECKPOINT_DIR, "best_expert_flair.pth"),
}

RESULT_DIR = os.path.join(DRIVE_PROJECT_ROOT, "results")


# =========================================================================
# 2. Data Splitting & Stratification Configuration
# =========================================================================
# Fixed seed for exact reproducibility across sessions
RANDOM_SEED = 73

# 80/10/10 data split ratios
# TRAIN_RATIO = 0.80
# VAL_RATIO = 0.10
# TEST_RATIO = 0.10


# =========================================================================
# 3. Model Hyperparameters & Input Dimensions
# =========================================================================
# Structural modalities and sequence order
MODALITIES = ["t1", "t1ce", "t2", "flair"]
NUM_MODALITIES = len(MODALITIES)

# Unique target channels after mapping label 4 to 3
NUM_SEG_CLASSES = 4

# Sub-volume dimensions for 3D patch training
PATCH_SIZE = (64, 64, 64)

# Dimensionality of the token embeddings across the Mamba blocks
EMBED_DIM = 96


# =========================================================================
# 4. Training Loop Optimization Settings
# =========================================================================
# Batch size per step
BATCH_SIZE = 4

# Total targeted training epochs across each session
NUM_EPOCHS = 90

# Stage 1: Expert Pre-Training Settings
PRETRAIN_EPOCHS = 300
PRETRAIN_LR = 1e-3

# Stage 2: Joint Training Settings
TOTAL_EPOCHS = 300
BASE_LR = 1e-3
ENCODER_LR = 1e-4  # Lower LR for fine-tuning pre-trained encoders
WEIGHT_DECAY = 1e-5

# Lowest learning rate boundary for the scheduler decay cycle
# ETA_MIN = 1e-6 # required only for CosineAnnealingWarmRestarts scheduler

# Safe RAM cache threshold for MONAI CacheDataset on Free Colab
CACHE_RATE = 0.0  # was 0.3 earlier (session crashed due to RAM overload)
NUM_WORKERS = 2


# =========================================================================
# 5. Advanced Component Parameters
# =========================================================================
# Epoch threshold to unfreeze encoders (Phase 1 Frozen Warm-up)
FROZEN_WARMUP_EPOCHS = 30

# Weight multiplier for the shared-weight auxiliary decoder loss
AUX_LOSS_WEIGHT = 0.4