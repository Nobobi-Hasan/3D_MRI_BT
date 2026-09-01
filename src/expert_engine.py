# src/expert_engine.py

import os
import csv
import gc  # Added for garbage collection
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from monai.inferers import sliding_window_inference
import src.config as config
from src.metrics import SegmentationMetrics

def train_one_epoch(model_components, dataloader, criterion, optimizer, scaler, device, target_modality_idx):
    """Trains the 4 isolated single-modality architectural components for one epoch."""
    conv_stem, patch_embed, mamba_backbone, aux_decoder = model_components
    
    conv_stem.train()
    patch_embed.train()
    mamba_backbone.train()
    aux_decoder.train()

    running_loss = 0.0
    running_seg_loss = 0.0

    for batch in tqdm(dataloader, desc="Training Batches", leave=False):
        images = batch["image"].to(device)
        seg_targets = batch["label"].to(device)
        
        # Isolate the specific modality channel for this expert (Shape: B, 1, H, W, D)
        mod_channel = images[:, target_modality_idx:target_modality_idx+1, :, :, :]
        
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # 1. Forward pass through unimodal feature stem
            feat1, feat2, feat3 = conv_stem(mod_channel)
            
            # 2. Tokenize the lowest resolution spatial map
            tokens, spatial_shape = patch_embed(feat3)
            
            # 3. Process tokens through BiMamba sequential blocks
            encoded_tokens = mamba_backbone(tokens)
            
            # 4. Decode representations back into 3D segmentation map using skip connections
            skip_features = [feat1, feat2, feat3]
            seg_logits = aux_decoder(encoded_tokens, spatial_shape, skip_features)
            
            # 5. Calculate Loss (DiceCE)
            # Pass None for aux_preds since this is a single isolated pathway
            loss_seg = criterion(seg_logits, seg_targets, aux_preds=None)

        scaler.scale(loss_seg).backward()
        scaler.step(optimizer)
        scaler.update()

        running_seg_loss += loss_seg.item()

    num_batches = len(dataloader)
    return running_seg_loss / num_batches


@torch.no_grad()
def validate_one_epoch(model_components, dataloader, criterion, device, target_modality_idx):
    """Evaluates the isolated expert pipeline on validation subsets with ground-truth masks."""
    conv_stem, patch_embed, mamba_backbone, aux_decoder = model_components
    
    conv_stem.eval()
    patch_embed.eval()
    mamba_backbone.eval()
    aux_decoder.eval()

    running_loss = 0.0
    seg_tracker = SegmentationMetrics()
    
    for batch in tqdm(dataloader, desc="Validation Batches", leave=False):
        images = batch["image"].to(device)
        seg_targets = batch["label"].to(device)

        B_current = images.size(0)
        batch_seg_logits = []

        # Iterate through batch elements individually to align localized sliding window metrics
        for b in range(B_current):
            # Isolate single volume and specific modality channel
            single_img = images[b:b+1, target_modality_idx:target_modality_idx+1, :, :, :]  # Shape: (1, 1, 128, 128, 128)

            def evaluation_predictor(patch_images):
                # Sequential forward pass for sliding window patches
                feat1, feat2, feat3 = conv_stem(patch_images)
                tokens, spatial_shape = patch_embed(feat3)
                encoded_tokens = mamba_backbone(tokens)
                skip_features = [feat1, feat2, feat3]
                seg_logits = aux_decoder(encoded_tokens, spatial_shape, skip_features)

                return seg_logits

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # Perform sliding window inference over a single validation volume to isolate feature scales
                seg_logits = sliding_window_inference(
                    inputs=single_img,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=16,
                    predictor=evaluation_predictor,
                    overlap=0.5,
                    mode="gaussian"
                )
            
            batch_seg_logits.append(seg_logits)

        # Re-assemble the individual predictions back to match original batch shapes
        seg_logits = torch.cat(batch_seg_logits, dim=0)  # Shape: (B, 4, 128, 128, 128)
        
        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            loss_seg = criterion(seg_logits, seg_targets, aux_preds=None)

        running_loss += loss_seg.item()

        seg_preds = torch.argmax(seg_logits, dim=1, keepdim=True)
        seg_tracker.update(seg_preds, seg_targets, run_hd=False)

    metrics = seg_tracker.compute(run_hd=False)

    metrics["val_loss"] = running_loss / len(dataloader)
    
    # Clear the tracker and force Python Garbage Collection to prevent System RAM leaks
    if hasattr(seg_tracker, 'reset'):
        seg_tracker.reset()
    del seg_tracker
    gc.collect()

    return metrics


def run_training(model_components, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, target_modality_idx, save_path):
    
    os.makedirs(config.EXPERT_CHECKPOINT_DIR, exist_ok=True)
    modality_name = config.MODALITIES[target_modality_idx]
    
    latest_path = os.path.join(config.EXPERT_CHECKPOINT_DIR, f"latest_expert_{modality_name}.pth")
    best_seg_path = save_path

    # --- Setup CSV Logging Directory and File ---
    results_dir = os.path.join(config.DRIVE_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, f"training_metrics_expert_{modality_name}.csv")
    # ------------------------------------------

    start_epoch = 0
    best_mean_dice = 0.0

    if os.path.exists(latest_path):
        print(f"[*] Found existing checkpoint record at: {latest_path}. Loading state...")
        checkpoint = torch.load(latest_path, map_location=device)
        
        model_components[0].load_state_dict(checkpoint["conv_stem_state"])
        model_components[1].load_state_dict(checkpoint["patch_embed_state"])
        model_components[2].load_state_dict(checkpoint["mamba_backbone_state"])
        model_components[3].load_state_dict(checkpoint["aux_decoder_state"])
        
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if scheduler and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            
        scaler.load_state_dict(checkpoint["scaler_state"])
        
        start_epoch = checkpoint["epoch"]
        best_mean_dice = checkpoint.get("best_mean_dice", 0.0)
        print(f"[+] Recovery complete. Resuming from absolute internal epoch counter: {start_epoch}")
    else:
        print(f"[*] No prior checkpoint found for {modality_name.upper()}. Initializing a new training.")

    # --- Initialize CSV Header if starting fresh ---
    if start_epoch == 0:
        if os.path.exists(csv_file):
            os.remove(csv_file)
            
    if start_epoch == 0 or not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch Number", "[Train] Seg Loss", "Mean Dice", "WT Dice", "TC Dice", "ET Dice", "Best"])
    # -----------------------------------------------

    target_epoch = start_epoch + config.PRETRAIN_EPOCHS
    print(f"[*] Incremental Run Configuration: Training from Epoch {start_epoch} -> Target Epoch {target_epoch} (+{config.PRETRAIN_EPOCHS} epochs)")

    for epoch in range(start_epoch, target_epoch):
        print(f"\n--- Epoch {epoch + 1}/{target_epoch} ---")
        
        train_seg_loss = train_one_epoch(
            model_components, train_loader, criterion, optimizer, scaler, device, target_modality_idx
        )
        print(f"[Train] Seg Loss: {train_seg_loss:.4f}")

        val_metrics = validate_one_epoch(
            model_components, val_loader, criterion, device, target_modality_idx
        )
        
        # Calculate Segmentation metrics
        mean_dice = (val_metrics["dice_WT"] + val_metrics["dice_TC"] + val_metrics["dice_ET"]) / 3.0
        
        print(f"[Val] Segmentation Loss-> Mean Dice: {mean_dice:.4f} (WT: {val_metrics['dice_WT']:.4f}, TC: {val_metrics['dice_TC']:.4f}, ET: {val_metrics['dice_ET']:.4f})")

        if scheduler:
            scheduler.step()

        # Check if the current epoch is the best one for CSV logging
        is_best = "YES" if mean_dice > best_mean_dice else "NO"

        # Update historical threshold metrics safely
        current_best_mean_dice = max(mean_dice, best_mean_dice)

        # Dictionary explicitly mapped for the joint training MambaBackbone loader function
        checkpoint_state = {
            "epoch": epoch + 1,
            "conv_stem_state": model_components[0].state_dict(),
            "patch_embed_state": model_components[1].state_dict(),
            "mamba_backbone_state": model_components[2].state_dict(),
            "aux_decoder_state": model_components[3].state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict(),
            "dice_WT": val_metrics["dice_WT"],
            "dice_TC": val_metrics["dice_TC"],
            "dice_ET": val_metrics["dice_ET"],
            "mean_dice": mean_dice,
            "best_mean_dice": current_best_mean_dice,
        }

        # Save Latest Progress Checkpoint immediately after every single epoch loop completes
        torch.save(checkpoint_state, latest_path)
        print(f"Stateful tracking saved to: {latest_path}")

        # 1. Evaluate and track Independent Peak Segmentation Weights
        if mean_dice > best_mean_dice:
            best_mean_dice = mean_dice
            torch.save(checkpoint_state, best_seg_path)
            print(f"*** best expert framework model configuration stored at: {best_seg_path}")

        # --- Append metrics to CSV ---
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{train_seg_loss:.4f}", 
                f"{mean_dice:.4f}", 
                f"{val_metrics['dice_WT']:.4f}", 
                f"{val_metrics['dice_TC']:.4f}", 
                f"{val_metrics['dice_ET']:.4f}", 
                is_best
            ])
        # -----------------------------
        
        # Clear GPU memory fragmentation safely at the end of each complete epoch cycle
        torch.cuda.empty_cache()

    print(f"\n Incremental cycle finished successfully. Total absolute epochs processed: {target_epoch}")