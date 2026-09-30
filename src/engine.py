# src/engine.py

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

def train_one_epoch(model_components, dataloader, criterion, optimizer, scaler, device, epoch):
    """Trains all 5 architectural model components concurrently for one epoch."""
    backbone, fusion, shared_backbone, decoder, aux_decoder = model_components
    
    backbone.train()
    fusion.train()
    shared_backbone.train()
    decoder.train()
    aux_decoder.train()

    running_loss = 0.0
    running_seg_loss = 0.0

    for batch in tqdm(dataloader, desc="Training Batches", leave=False):
        images = batch["image"].to(device)
        seg_targets = batch["label"].to(device)
        
        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type=device.type, enabled=(device.type == "cuda")):
            # 1. Forward pass through backbone
            # Missing modalities (zeroed by RandDropModalityd) are automatically skipped inside MambaBackbone
            modality_tokens, spatial_shape, skip_features, single_skip_features = backbone(images)
            
            # 2. Main Pathway (Pathway B): Process fused features
            # Fusion block detects missing channels dynamically from 'images' and injects absent_emb
            fused_tokens = fusion(modality_tokens, images)
            latent_tokens = shared_backbone(fused_tokens)
            
            # Feed the skip features to the main decoder
            seg_logits = decoder(latent_tokens, spatial_shape, skip_features)
            
            # 3. Auxiliary Pathway (Pathway A): Independent supervision on strictly present modalities
            aux_preds = []
            B_current, num_mods = images.shape[0], len(modality_tokens)
            
            # Dynamically detect missing modalities by checking if channels equal 0
            presence = (images.reshape(B_current, num_mods, -1).abs().sum(dim=-1) > 1e-5)
            
            for idx in range(num_mods):
                # ONLY compute auxiliary forward pass if the modality is present
                if presence[:, idx].any():
                    mod_skips = [single_skip_features[0][idx], single_skip_features[1][idx], single_skip_features[2][idx]]
                    aux_pred = aux_decoder(modality_tokens[idx], spatial_shape, mod_skips)
                    aux_preds.append(aux_pred)

            # 4. Calculate Combined Loss (DiceCE + Scaled Aux DiceCE)
            loss_seg = criterion(seg_logits, seg_targets, aux_preds)

        scaler.scale(loss_seg).backward()
        scaler.step(optimizer)
        scaler.update()

        running_seg_loss += loss_seg.item()

    num_batches = len(dataloader)
    return running_seg_loss / num_batches


@torch.no_grad()
def validate_one_epoch(model_components, dataloader, criterion, device):
    """Evaluates all 5 components on validation subsets across all 15 missing-modality combinations."""
    backbone, fusion, shared_backbone, decoder, aux_decoder = model_components
    
    backbone.eval()
    fusion.eval()
    shared_backbone.eval()
    decoder.eval()
    aux_decoder.eval()

    running_loss = 0.0
    seg_tracker = SegmentationMetrics()
    
    combinations = config.POSSIBLE_DROPPED_MODALITY_COMBINATIONS
    
    for batch in tqdm(dataloader, desc="Validation Batches (15 Combinations)", leave=False):
        images_original = batch["image"].to(device)
        seg_targets = batch["label"].to(device)
        B_current = images_original.size(0)

        # Loop through all 15 missing modality scenarios for the current batch
        for comb in combinations:
            masked_images = images_original.clone()
            
            # Apply deterministic zero-mask for the current missing-modality combination
            for idx in comb:
                masked_images[:, idx, ...] = 0.0

            batch_seg_logits = []

            # Iterate through batch elements individually to align localized sliding window metrics
            for b in range(B_current):
                single_img = masked_images[b:b+1]  # Shape: (1, 4, 128, 128, 128)

                def evaluation_predictor(patch_images):
                    # Unpack the updated 4 outputs from the backbone
                    modality_tokens, spatial_shape, skip_features, _ = backbone(patch_images)
                    fused_tokens = fusion(modality_tokens, patch_images)
                    latent_tokens = shared_backbone(fused_tokens)
                    seg_logits = decoder(latent_tokens, spatial_shape, skip_features)
                    return seg_logits

                with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                    # Perform sliding window inference over a single validation volume
                    seg_logits = sliding_window_inference(
                        inputs=single_img,
                        roi_size=config.PATCH_SIZE,
                        sw_batch_size=24,
                        predictor=evaluation_predictor,
                        overlap=0.5,
                        mode="gaussian"
                    )
                
                batch_seg_logits.append(seg_logits)

            # Re-assemble the individual predictions back to match original batch shapes
            seg_logits = torch.cat(batch_seg_logits, dim=0)  # Shape: (B, 4, 128, 128, 128)
            
            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # Pass None for aux_preds since we do not calculate auxiliary loss during validation
                loss_seg = criterion(seg_logits, seg_targets, aux_preds=None)

            running_loss += loss_seg.item()

            seg_preds = torch.argmax(seg_logits, dim=1, keepdim=True)
            # Accumulate metrics across all 15 combinations natively
            seg_tracker.update(seg_preds, seg_targets, run_hd=False)

    metrics = seg_tracker.compute(run_hd=False)

    # Adjust the loss denominator to account for the 15 combination runs per batch
    metrics["val_loss"] = running_loss / (len(dataloader) * len(combinations))
    
    # Clear the tracker and force Python Garbage Collection to prevent System RAM leaks
    if hasattr(seg_tracker, 'reset'):
        seg_tracker.reset()
    del seg_tracker
    gc.collect()

    return metrics


def run_training(model_components, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device, session_epoch = config.NUM_EPOCHS):
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    latest_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
    best_seg_path = os.path.join(config.CHECKPOINT_DIR, "best_seg.pth")

    # --- Setup CSV Logging Directory and File ---
    results_dir = os.path.join(config.DRIVE_PROJECT_ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, "training_metrics.csv")
    # ------------------------------------------

    start_epoch = 0
    best_mean_dice = 0.0
    last_val_wt = 0.0
    last_val_tc = 0.0
    last_val_et = 0.0
    last_mean_dice = 0.0

    if os.path.exists(latest_path):
        print(f"[*] Found existing checkpoint record at: {latest_path}. Loading state...")
        checkpoint = torch.load(latest_path, map_location=device)
        
        model_components[0].load_state_dict(checkpoint["backbone_state"])
        model_components[1].load_state_dict(checkpoint["fusion_state"])
        model_components[2].load_state_dict(checkpoint["shared_backbone_state"])
        model_components[3].load_state_dict(checkpoint["decoder_state"])
        model_components[4].load_state_dict(checkpoint["aux_decoder_state"])
        
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        
        if scheduler and checkpoint.get("scheduler_state") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            
        scaler.load_state_dict(checkpoint["scaler_state"])
        
        start_epoch = checkpoint["epoch"]
        best_mean_dice = checkpoint.get("best_mean_dice", 0.0)
        last_val_wt = checkpoint.get("dice_WT", 0.0)
        last_val_tc = checkpoint.get("dice_TC", 0.0)
        last_val_et = checkpoint.get("dice_ET", 0.0)
        last_mean_dice = checkpoint.get("mean_dice", 0.0)
        print(f"[+] Recovery complete. Resuming from absolute internal epoch counter: {start_epoch}")
    else:
        print("[*] No prior checkpoint found. Initializing a new training.")

    # --- Initialize CSV Header if starting fresh ---
    if start_epoch == 0:
        if os.path.exists(csv_file):
            os.remove(csv_file)
            
    if start_epoch == 0 or not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Epoch Number", "[Train] Seg Loss", "Mean Dice", "WT Dice", "TC Dice", "ET Dice", "Best"])
    # -----------------------------------------------

    target_epoch = start_epoch + session_epoch
    print(f"[*] Incremental Run Configuration: Training from Epoch {start_epoch} -> Target Epoch {target_epoch} (+{session_epoch} epochs)")

    for epoch in range(start_epoch, target_epoch):
        current_epoch_num = epoch + 1
        print(f"\n--- Epoch {current_epoch_num}/{target_epoch} ---")
        
        # Phase 1 & 2: Frozen Warm-Up and Differential Fine-Tuning
        if epoch < config.FROZEN_WARMUP_EPOCHS:
            for param in model_components[0].parameters():  # model_components[0] is backbone
                param.requires_grad = False
        else:
            for param in model_components[0].parameters():
                param.requires_grad = True
        
        # Pass the current epoch integer to control the warmup/dropout logic
        train_seg_loss = train_one_epoch(
            model_components, train_loader, criterion, optimizer, scaler, device, epoch
        )
        print(f"[Train] Seg Loss: {train_seg_loss:.4f}")

        if scheduler:
            scheduler.step()

        # Determine whether to run validation (Every 5 epochs, matching SimMLM, or on final target epoch)
        run_val = (current_epoch_num % 5 == 0) or (current_epoch_num == target_epoch)
        is_best = "NO"

        if run_val:
            val_metrics = validate_one_epoch(model_components, val_loader, criterion, device)
            
            # Calculate Segmentation metrics
            mean_dice = (val_metrics["dice_WT"] + val_metrics["dice_TC"] + val_metrics["dice_ET"]) / 3.0
            
            last_val_wt = val_metrics["dice_WT"]
            last_val_tc = val_metrics["dice_TC"]
            last_val_et = val_metrics["dice_ET"]
            last_mean_dice = mean_dice

            print(f"[Val] Segmentation Loss-> Mean Dice: {mean_dice:.4f} (WT: {last_val_wt:.4f}, TC: {last_val_tc:.4f}, ET: {last_val_et:.4f})")

            # Check if the current validation evaluation produces a new peak performance
            if mean_dice > best_mean_dice:
                best_mean_dice = mean_dice
                is_best = "YES"

            # Update historical threshold metrics safely
            current_best_mean_dice = best_mean_dice
        else:
            print(f"[Val] Skipped for Epoch {current_epoch_num} (Validating every 5 epochs).")
            current_best_mean_dice = best_mean_dice

        checkpoint_state = {
            "epoch": current_epoch_num,
            "backbone_state": model_components[0].state_dict(),
            "fusion_state": model_components[1].state_dict(),
            "shared_backbone_state": model_components[2].state_dict(),
            "decoder_state": model_components[3].state_dict(),
            "aux_decoder_state": model_components[4].state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "scaler_state": scaler.state_dict(),
            "dice_WT": last_val_wt,
            "dice_TC": last_val_tc,
            "dice_ET": last_val_et,
            "mean_dice": last_mean_dice,
            "best_mean_dice": current_best_mean_dice,
        }

        # Save Latest Progress Checkpoint immediately after every single epoch loop completes
        torch.save(checkpoint_state, latest_path)
        print(f"Stateful tracking saved to: {latest_path}")

        # 1. Evaluate and track Independent Peak Segmentation Weights
        if run_val and is_best == "YES":
            torch.save(checkpoint_state, best_seg_path)
            print(f"*** best segmentation framework model configuration stored at: {best_seg_path}")

        # --- Append metrics to CSV ---
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_epoch_num, 
                f"{train_seg_loss:.4f}", 
                f"{last_mean_dice:.4f}" if run_val else "-", 
                f"{last_val_wt:.4f}" if run_val else "-", 
                f"{last_val_tc:.4f}" if run_val else "-", 
                f"{last_val_et:.4f}" if run_val else "-", 
                is_best
            ])
        # -----------------------------
        
        # Clear GPU memory fragmentation safely at the end of each complete epoch cycle
        torch.cuda.empty_cache()

    print(f"\n Incremental cycle finished successfully. Total absolute epochs processed: {target_epoch}")