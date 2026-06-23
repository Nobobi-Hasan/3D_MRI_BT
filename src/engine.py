# src/engine.py

import os
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
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
            modality_tokens, spatial_shape = backbone(images)
            fused_tokens = fusion(modality_tokens, images)
            latent_tokens = shared_backbone(fused_tokens)
            seg_logits, refined_features = decoder(latent_tokens, spatial_shape)
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

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            modality_tokens, spatial_shape = backbone(images)
            fused_tokens = fusion(modality_tokens, images)
            latent_tokens = shared_backbone(fused_tokens)
            seg_logits, refined_features = decoder(latent_tokens, spatial_shape)
            cls_logits, _ = classifier(seg_logits, refined_features, gt_labels=None)
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
    best_path = os.path.join(config.CHECKPOINT_DIR, "best_checkpoint.pth")

    start_epoch = 0
    best_combined_score = 0.0  # UPDATED: Tracks blended metric equilibrium

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
        
        # Calculate Multi-Task balanced score validation checkpoint indicators
        mean_dice = (val_metrics["dice_WT"] + val_metrics["dice_TC"] + val_metrics["dice_ET"]) / 3.0
        combined_score = (0.5 * mean_dice) + (0.5 * val_metrics["macro_f1"])
        
        print(f"[Val] Loss: {val_metrics['val_loss']:.4f} | Mean Dice: {mean_dice:.4f} | Macro F1: {val_metrics['macro_f1']:.4f} | Combined Score: {combined_score:.4f}")

        if scheduler:
            scheduler.step()

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
            "best_combined_score": max(combined_score, best_combined_score),
        }

        torch.save(checkpoint_state, latest_path)
        print(f"Stateful tracking saved to: {latest_path}")

        # Save best multi-task framework model configuration
        if combined_score > best_combined_score:
            best_combined_score = combined_score
            torch.save(checkpoint_state, best_path)
            print(f"*** best multi-task framework model configuration stored at: {best_path}")

    print(f"\n Incremental cycle finished successfully. Total absolute epochs processed: {target_epoch}")