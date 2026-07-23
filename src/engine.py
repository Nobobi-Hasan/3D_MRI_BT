# src/engine.py

import os
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
            # 1. Forward pass through backbone to get standard and single-modality features
            modality_tokens, spatial_shape, skip_features, single_skip_features = backbone(images)
            
            # --- Phase 5: Pseudo-Curriculum Warmup & Token-Level Feature Dropout ---
            # No dropout during the warmup phase to establish stable initial representations
            current_dropout_rate = 0.0 if epoch < config.WARMUP_EPOCHS else config.MODALITY_DROPOUT_PROB
            
            num_mods = len(modality_tokens)
            active_modalities = []
            processed_modality_tokens = []
            
            # Create a per-batch mask for modality dropping
            keep_mask = torch.ones(num_mods, device=device)
            if current_dropout_rate > 0.0:
                keep_mask = (torch.rand(num_mods, device=device) > current_dropout_rate).float()
                # Fallback: if all modalities drop, randomly keep one to prevent zero-gradient failure
                if keep_mask.sum() == 0:
                    keep_mask[torch.randint(0, num_mods, (1,)).item()] = 1.0
            
            processed_images = images.clone()
            for i in range(num_mods):
                if keep_mask[i] == 1.0:
                    processed_modality_tokens.append(modality_tokens[i])
                    active_modalities.append(i)
                else:
                    # Replace dropped modality tokens and original image channels with zeros
                    processed_modality_tokens.append(torch.zeros_like(modality_tokens[i]))
                    processed_images[:, i, :, :, :] = 0.0
            # -----------------------------------------------------------------------

            # 2. Main Pathway (Pathway B): Process fused masked features
            fused_tokens = fusion(processed_modality_tokens, processed_images)
            latent_tokens = shared_backbone(fused_tokens)
            seg_logits = decoder(latent_tokens, spatial_shape, skip_features)
            
            # 3. Auxiliary Pathway (Pathway A): Independent supervision on unmasked modalities
            aux_preds = []
            for idx in active_modalities:
                mod_skips = [single_skip_features[0][idx], single_skip_features[1][idx]]
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
    """Evaluates all 5 components on validation subsets with ground-truth masks."""
    backbone, fusion, shared_backbone, decoder, aux_decoder = model_components
    
    backbone.eval()
    fusion.eval()
    shared_backbone.eval()
    decoder.eval()
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
            single_img = images[b:b+1]  # Shape: (1, 4, 128, 128, 128)

            def evaluation_predictor(patch_images):
                # Unpack the updated 4 outputs from the backbone (ignore single_skip_features for validation)
                modality_tokens, spatial_shape, skip_features, _ = backbone(patch_images)
                # No dropout applied during validation phase
                fused_tokens = fusion(modality_tokens, patch_images)
                latent_tokens = shared_backbone(fused_tokens)
                # Unpack only logits since classification refined_features were removed
                seg_logits = decoder(latent_tokens, spatial_shape, skip_features)

                return seg_logits

            with autocast(device_type=device.type, enabled=(device.type == "cuda")):
                # Perform sliding window inference over a single validation volume to isolate feature scales
                seg_logits = sliding_window_inference(
                    inputs=single_img,
                    roi_size=config.PATCH_SIZE,
                    sw_batch_size=1,
                    predictor=evaluation_predictor,
                    overlap=0.25,
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
        seg_tracker.update(seg_preds, seg_targets, run_hd=False)

    metrics = seg_tracker.compute(run_hd=False)

    metrics["val_loss"] = running_loss / len(dataloader)

    return metrics


def run_training(model_components, train_loader, val_loader, criterion, optimizer, scheduler, scaler, device):
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    latest_path = os.path.join(config.CHECKPOINT_DIR, "latest_checkpoint.pth")
    best_seg_path = os.path.join(config.CHECKPOINT_DIR, "best_seg.pth")

    start_epoch = 0
    best_mean_dice = 0.0

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
        print(f"[+] Recovery complete. Resuming from absolute internal epoch counter: {start_epoch}")
    else:
        print("[*] No prior checkpoint found. Initializing a new training.")

    target_epoch = start_epoch + config.NUM_EPOCHS
    print(f"[*] Incremental Run Configuration: Training from Epoch {start_epoch} -> Target Epoch {target_epoch} (+{config.NUM_EPOCHS} epochs)")

    for epoch in range(start_epoch, target_epoch):
        print(f"\n--- Epoch {epoch + 1}/{target_epoch} ---")
        
        # Pass the current epoch integer to control the warmup/dropout logic
        train_seg_loss = train_one_epoch(
            model_components, train_loader, criterion, optimizer, scaler, device, epoch
        )
        print(f"[Train] Seg Loss: {train_seg_loss:.4f}")

        val_metrics = validate_one_epoch(model_components, val_loader, criterion, device)
        
        # Calculate Segmentation metrics
        mean_dice = (val_metrics["dice_WT"] + val_metrics["dice_TC"] + val_metrics["dice_ET"]) / 3.0
        
        print(f"[Val] Segmentation Loss-> Mean Dice: {mean_dice:.4f} (WT: {val_metrics['dice_WT']:.4f}, TC: {val_metrics['dice_TC']:.4f}, ET: {val_metrics['dice_ET']:.4f})")


        if scheduler:
            scheduler.step()

        # Update historical threshold metrics safely
        current_best_mean_dice = max(mean_dice, best_mean_dice)

        checkpoint_state = {
            "epoch": epoch + 1,
            "backbone_state": model_components[0].state_dict(),
            "fusion_state": model_components[1].state_dict(),
            "shared_backbone_state": model_components[2].state_dict(),
            "decoder_state": model_components[3].state_dict(),
            "aux_decoder_state": model_components[4].state_dict(),
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
            print(f"*** best segmentation framework model configuration stored at: {best_seg_path}")

    print(f"\n Incremental cycle finished successfully. Total absolute epochs processed: {target_epoch}")