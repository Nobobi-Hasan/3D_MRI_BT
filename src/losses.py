# src/losses.py

import torch
import torch.nn as nn
from monai.losses import DiceLoss
import src.config as config

class SegmentationLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # MONAI Dice Loss for segmentation
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)

    def forward(self, seg_pred, seg_target):
        # compute segmentation Dice loss
        loss_seg = self.dice_loss(seg_pred, seg_target)
        
        return loss_seg