# src/engine.py

import os
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import src.config as config
from src.metrics import SegmentationMetrics, compute_classification_metrics

def train_one_epoch(model_components, dataloader, criterion, optimizer, scaler, device):
    """Trains all 5 architectural model components concurrently for one epoch."""
    backbone, fusion, shared_backbone, decoder, classifier = model_components
    
    backbone.train()
    fusion.train()
    shared_backbone.train()
    decoder.train()
    classifier.train()

    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0

    for batch in tqdm(dataloader, desc="Training Batches", leave=False):
        images = batch["image"].to(device)
        seg_targets = batch["label"].to(device)
        cls_targets = batch["grade"].to(device)
        
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            modality_tokens, spatial_shape, skip_features = backbone(images)
            fused_tokens = fusion(modality_tokens, images)
            latent_tokens = shared_backbone(fused_tokens)
            seg_logits, refined_features = decoder(latent_tokens, spatial_shape, skip_features)
            cls_logits, _ = classifier(seg_logits, refined_features, gt_labels=seg_targets)
            total_loss, loss_seg, loss_cls = criterion(seg_logits, seg_targets, cls_logits, cls_targets)

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += total_loss.item()
        running_seg_loss += loss_seg.item()
        running_cls_loss += loss_cls.item()

    num_batches = len(dataloader)
    return running_loss / num_batches, running_seg_loss / num_batches, running_cls_loss / num_batches


@torch.no_grad()
def validate_one_epoch(model_components, dataloader, criterion, device):
    """Evaluates all 5 components on validation subsets with ground-truth masks."""
    backbone, fusion, shared_backbone, decoder, classifier = model_components
    
    backbone.eval()
    fusion.eval()
    shared_backbone.eval()
    decoder.eval()
    classifier.eval()

    running_loss = 0.0
    seg_tracker = SegmentationMetrics()
    
    all_cls_preds = []
    all_cls_targets = []
    all_cls_probs = []

    for batch in tqdm(dataloader, desc="Validation Batches", leave=False):
        images = batch["image"].to(device)
        seg_targets = batch["label"].to(device)
        cls_targets = batch["grade"].to(device)

        # Volumetric evaluation patch collector to compute combined multi-task metrics safely
        volume_cls_logits = []

        def evaluation_predictor(patch_images):
            modality_tokens, spatial_shape, skip_features = backbone(patch_images)
            fused_tokens = fusion(modality_tokens, patch_images)
            latent_tokens = shared_backbone(fused_tokens)
            seg_logits, refined_features = decoder(latent_tokens, spatial_shape, skip_features)
            
            p_cls_logits, _ = classifier(seg_logits, refined_features, gt_labels=None)
            volume_cls_logits.append(p_cls_logits)
            return seg_logits

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # Perform sliding window inference over full validation volumes to protect VRAM metrics profiles
            seg_logits = sliding_window_inference(
                inputs=images,
                roi_size=config.PATCH_SIZE,
                sw_batch_size=1,
                predictor=evaluation_predictor,
                overlap=0.25,
                mode="gaussian"
            )
            
            # Aggregate patch-level classification representations back into full macro metrics context
            cls_logits = torch.stack(volume_cls_logits, dim=0).mean(dim=0)
            total_loss, _, _ = criterion(seg_logits, seg_targets, cls_logits, cls_targets)

        running_loss += total_loss.item()

        seg_preds = torch.argmax(seg_logits, dim=1, keepdim=True)
        seg_tracker.update(seg_preds, seg_targets, run_hd=False)

        cls_probs = F.softmax(cls_logits, dim=1)
        cls_preds = torch.argmax(cls_logits, dim=1)

        all_cls_preds.append(cls_preds)
        all_cls_targets.append(cls_targets)
        all_cls_probs.append(cls_probs)

    metrics = seg_tracker.compute(run_hd=False)
    
    all_cls_preds = torch.cat(all_cls_preds, dim=0)
    all_cls_targets = torch.cat(all_cls_targets, dim=0)
    all_cls_probs = torch.cat(all_cls_probs, dim=0)
    
    cls_metrics = compute_classification_metrics(all_cls_preds, all_cls_targets, all_cls_probs)
    metrics.update(cls_metrics)
    metrics["val_loss"] = running_loss / len(dataloader)

    return metrics


def run_training(model_components, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device):
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    latest_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
    best_seg_path = os.path.join(config.CHECKPOINT_DIR, "best_seg.pth")
    best_cls_path = os.path.join(config.CHECKPOINT_DIR, "best_cls.pth")
    best_multitask_path = os.path.join(config.CHECKPOINT_DIR, "best_multitask.pth")

    start_epoch = 0
    best_mean_dice = 0.0
    best_macro_f1 = 0.0
    best_combined_score = 0.0

    if os.path.exists(latest_path):
        print(f"[*] Found existing checkpoint record at: {latest_path}. Loading state...")
        checkpoint = torch.load(latest_path, map_location=device)
        
        model_components[0].load_state_dict(checkpoint["backbone_state"])
        model_components[1].load_state_dict(checkpoint["fusion_state"])
        model_components[2].load_state_dict(checkpoint["shared_backbone_state"])
        model_components[3].load_state_dict(checkpoint["decoder_state"])
        model_components[4].load_state_dict(checkpoint["classifier_state"])
        
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if scheduler and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            
        scaler.load_state_dict(checkpoint["scaler_state"])
        
        start_epoch = checkpoint["epoch"]
        best_mean_dice = checkpoint.get("best_mean_dice", 0.0)
        best_macro_f1 = checkpoint.get("best_macro_f1", 0.0)
        best_combined_score = checkpoint.get("best_combined_score", 0.0)
        print(f"[+] Recovery complete. Resuming from absolute internal epoch counter: {start_epoch}")
    else:
        print("[*] No prior checkpoint found. Initializing a new training.")

    target_epoch = start_epoch + config.NUM_EPOCHS
    print(f"[*] Incremental Run Configuration: Training from Epoch {start_epoch} -> Target Epoch {target_epoch} (+{config.NUM_EPOCHS} epochs)")

    for epoch in range(start_epoch, target_epoch):
        print(f"\n--- Epoch {epoch + 1}/{target_epoch} ---")
        
        train_loss, train_seg, train_cls = train_one_epoch(
            model_components, train_loader, criterion, optimizer, scaler, device
        )
        print(f"[Train] Total Loss: {train_loss:.4f} | Seg Loss: {train_seg:.4f} | Cls Loss: {train_cls:.4f}")

        val_metrics = validate_one_epoch(model_components, val_loader, criterion, device)
        
        # Calculate Multi-Task balanced score metrics
        mean_dice = (val_metrics["dice_WT"] + val_metrics["dice_TC"] + val_metrics["dice_ET"]) / 3.0
        combined_score = (0.4 * mean_dice) + (0.3 * val_metrics["macro_f1"]) + (0.3 * val_metrics["roc_auc"])
        
        print(f"[Val] Loss: {val_metrics['val_loss']:.4f} | Mean Dice: {mean_dice:.4f} | Macro F1: {val_metrics['macro_f1']:.4f} | ROC-AUC: {val_metrics['roc_auc']:.4f} | Combined Score: {combined_score:.4f}")

        if scheduler:
            scheduler.step()

        # Update historical threshold metrics safely
        current_best_mean_dice = max(mean_dice, best_mean_dice)
        current_best_macro_f1 = max(val_metrics["macro_f1"], best_macro_f1)
        current_best_combined_score = max(combined_score, best_combined_score)

        checkpoint_state = {
            "epoch": epoch + 1,
            "backbone_state": model_components[0].state_dict(),
            "fusion_state": model_components[1].state_dict(),
            "shared_backbone_state": model_components[2].state_dict(),
            "decoder_state": model_components[3].state_dict(),
            "classifier_state": model_components[4].state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict(),
            "mean_dice": mean_dice,
            "macro_f1": val_metrics["macro_f1"],
            "roc_auc": val_metrics["roc_auc"],
            "combined_score": combined_score,
            "best_mean_dice": current_best_mean_dice,
            "best_macro_f1": current_best_macro_f1,
            "best_combined_score": current_best_combined_score,
        }

        # Save Latest Progress Checkpoint immediately after every single epoch loop completes
        torch.save(checkpoint_state, latest_path)
        print(f"Stateful tracking saved to: {latest_path}")

        # 1. Evaluate and track Independent Peak Segmentation Weights
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(checkpoint_state, best_seg_path)
            print(f"*** best segmentation framework model configuration stored at: {best_seg_path}")

        # 2. Evaluate and track Independent Peak Classification Weights
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            torch.save(checkpoint_state, best_cls_path)
            print(f"*** best classification framework model configuration stored at: {best_cls_path}")

        # 3. Evaluate and track Optimal Blended Multi-Task Balanced Weights
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            torch.save(checkpoint_state, best_multitask_path)
            print(f"*** best multi-task framework model configuration stored at: {best_multitask_path}")

    print(f"\n Incremental cycle finished successfully. Total absolute epochs processed: {target_epoch}")