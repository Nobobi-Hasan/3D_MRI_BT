# src/losses.py

import torch
import torch.nn as nn
from monai.losses import DiceLoss
import src.config as config

class MultiTaskLoss(nn.Module):
    """Custom multi-task loss function bor both - segmentation and weighted classification."""
    def __init__(self):
        super().__init__()
        # MONAI Dice Loss for segmentation
        self.dice_loss = DiceLoss(to_onehot_y=True, softmax=True)
        
        # define tensor tracking for classification class weights to mitigate HGG/LGG imbalance
        self.register_buffer("class_weights", torch.tensor(config.CLASSIFICATION_CLASS_WEIGHTS, dtype=torch.float32))
        
        # Cross Entropy Loss for classification; using target balancing penalty weights
        self.cls_loss = nn.CrossEntropyLoss(weight=self.class_weights)

    def forward(self, seg_pred, seg_target, cls_pred, cls_target):
        # compute segmentation Dice loss
        loss_seg = self.dice_loss(seg_pred, seg_target)
        
        # compute stratified, class-weighted CE loss for classification
        loss_cls = self.cls_loss(cls_pred, cls_target)
        
        # combine both losses
        total_loss = loss_seg + loss_cls
        
        return total_loss, loss_seg, loss_cls