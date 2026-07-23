# src/losses.py

import torch
import torch.nn as nn
from monai.losses import DiceLoss
import src.config as config

class SegmentationLoss(nn.Module):
    def __init__(self, aux_weight=0.4):
        super().__init__()
        # MONAI DiceCE Loss combines Dice and Cross Entropy for better stability
        self.dice_ce_loss = DiceCELoss(to_onehot_y=True, softmax=True)
        # Weight for the shared-weight auxiliary decoder supervision
        self.aux_weight = getattr(config, 'AUX_LOSS_WEIGHT', aux_weight)

    def forward(self, seg_pred, seg_target, aux_preds=None):
        # compute main segmentation DiceCE loss
        loss_seg = self.dice_ce_loss(seg_pred, seg_target)
        
        # compute and add auxiliary segmentation DiceCE loss if predictions are provided
        if aux_preds is not None and len(aux_preds) > 0:
            aux_loss = 0.0
            for aux_pred in aux_preds:
                aux_loss = aux_loss + self.dice_ce_loss(aux_pred, seg_target)
            
            # average the auxiliary losses across the processed unmasked modalities
            aux_loss = aux_loss / len(aux_preds)
            
            # combine main loss with scaled auxiliary loss
            loss_seg = loss_seg + (self.aux_weight * aux_loss)
            
        return loss_seg