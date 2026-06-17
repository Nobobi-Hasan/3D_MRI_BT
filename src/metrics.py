# src/metrics.py

import torch
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from sklearn.metrics import f1_score, roc_auc_score

class SegmentationMetrics:
    """Accumulator for true 3D BraTS region evaluation metrics across batches."""
    def __init__(self):
        # initialize MONAI metric calculations for the 3 target clinical channels
        self.dice_metric = DiceMetric(include_background=True, reduction="mean_batch")
        self.hd_metric = HausdorffDistanceMetric(include_background=True, reduction="mean_batch")

    def _convert_to_brats_regions(self, y_pred, y):
        """Convert multiclass argmax tensors into overlapping BraTS clinical zones."""
        # I expect inputs to be spatial argmax labels of shape (B, 1, H, W, D)
        # Whole Tumor (WT) = Necrotic Core (1), + Edema (2) + Enhancing Tumor (3)
        pred_wt = (y_pred == 1) | (y_pred == 2) | (y_pred == 3)
        target_wt = (y == 1) | (y == 2) | (y == 3)
        
        # Tumor Core (TC) = Necrotic Core (1) + Enhancing Tumor (3)
        pred_tc = (y_pred == 1) | (y_pred == 3)
        target_tc = (y == 1) | (y == 3)
        
        # Enhancing Tumor (ET) = Enhancing Tumor (3)
        pred_et = (y_pred == 3)
        target_et = (y == 3)
        
        # stack the individual regions along the channel dimension to yield (B, 3, H, W, D)
        pred_regions = torch.cat([pred_wt, pred_tc, pred_et], dim=1).float()
        target_regions = torch.cat([target_wt, target_tc, target_et], dim=1).float()
        
        return pred_regions, target_regions

    def update(self, preds, targets, run_hd=False):
        # transform raw class predictions into overlapping clinical masks
        pred_regions, target_regions = self._convert_to_brats_regions(preds, targets)
        
        self.dice_metric(y_pred=pred_regions, y=target_regions)
        
        # only run the Hausdorff distance map calculations if run_hd = True
        if run_hd:
            self.hd_metric(y_pred=pred_regions, y=target_regions)

    def compute(self, run_hd=False):
        # Aggregate structural scores across the entire tracking epoch
        dice_vals = self.dice_metric.aggregate()
        self.dice_metric.reset()
        
        metrics_dict = {
            "dice_WT": dice_vals[0].item() if dice_vals.numel() > 0 else 0.0,
            "dice_TC": dice_vals[1].item() if dice_vals.numel() > 1 else 0.0,
            "dice_ET": dice_vals[2].item() if dice_vals.numel() > 2 else 0.0,
        }
        
        if run_hd:
            hd_vals = self.hd_metric.aggregate()
            self.hd_metric.reset()
            metrics_dict.update({
                "hd95_WT": hd_vals[0].item() if hd_vals.numel() > 0 else 0.0,
                "hd95_TC": hd_vals[1].item() if hd_vals.numel() > 1 else 0.0,
                "hd95_ET": hd_vals[2].item() if hd_vals.numel() > 2 else 0.0,
            })
        else:
            metrics_dict.update({
                "hd95_WT": 0.0,
                "hd95_TC": 0.0,
                "hd95_ET": 0.0,
            })
            
        return metrics_dict

def compute_classification_metrics(all_preds, all_targets, all_probs):
    """Compute macro-averaged and shape-safe classification metrics."""
    # to numpy arrays for compatibility with sklearn
    if isinstance(all_preds, torch.Tensor):
        all_preds = all_preds.cpu().numpy()
    if isinstance(all_targets, torch.Tensor):
        all_targets = all_targets.cpu().numpy()
    if isinstance(all_probs, torch.Tensor):
        all_probs = all_probs.cpu().numpy()
        
    # Verify the probability tracking array is flattened to a 1D structure
    if len(all_probs.shape) > 1 and all_probs.shape[1] == 2:
        all_probs = all_probs[:, 1]
        
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    
    try:
        auc = roc_auc_score(all_targets, all_probs)
    except ValueError:
        auc = 0.5
        
    return {
        "macro_f1": float(f1),
        "roc_auc": float(auc)
    }