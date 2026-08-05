# src/models/fusion.py

import torch
import torch.nn as nn

class PresenceAwareCrossModalFusion(nn.Module):
    """Phase 3: Stage 2 - Presence-Aware Cross-Modal Attention Fusion.
    Maps multimodal inputs and adjusts attention weights to isolate and
    compensate for missing modalities using learnable presence tokens.
    
    *Upgraded with IQ-RDA Block for O(1) spatial complexity and relational consistency.*
    """
    def __init__(self, embed_dim=96, num_heads=4, num_modalities=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_modalities = num_modalities

        # Phase 1.5: Learnable modality identity embeddings
        self.modality_embeds = nn.Parameter(torch.randn(num_modalities, 1, embed_dim) * 0.02)

        # Phase 1.5: Learnable binary presence/absence token encodings
        self.present_emb = nn.Parameter(torch.randn(num_modalities, 1, embed_dim) * 0.02)
        self.absent_emb = nn.Parameter(torch.randn(num_modalities, 1, embed_dim) * 0.02)

        # --- NEW: IQ-RDA Block Components ---
        # Step B: Intrinsic Quality Estimator (Q_i)
        self.iq_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )

        # Step C: Relational Consistency Estimator (\Delta R_i)
        self.rel_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.rel_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1)
        )
        # ------------------------------------

        # Channel-wise Concatenation + Projection 
        # Compresses concatenated features from (B, L, 384) back to (B, L, 96)
        # Assuming 4 modalities; and an embedding dimension of 96: 4 * 96 = 384
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim * num_modalities, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def forward(self, modality_tokens, x):
        """
        Args:
            modality_tokens (list of 4 Tensors): List containing [T1, T1ce, T2, FLAIR] 
                                                 each with shape (B, L, C)
            x (Tensor): Original raw 3D input tensor of shape (B, 4, H, W, D)
                        used to dynamically detect zeroed out dropped channels.
        Returns:
            Tensor: Unified Multi-Modal Feature Tensor of shape (B, L, C)
        """
        B, _, _, _, _ = x.shape
        L = modality_tokens[0].size(1)
        
        # FIX 1: Changed .view() to .reshape() to handle non-contiguous sliding window patches
        presence = (x.reshape(B, self.num_modalities, -1).abs().sum(dim=-1) > 1e-5).float()

        processed_tokens = []
        
        for i in range(self.num_modalities):
            tokens = modality_tokens[i] # Shape: (B, L, C)
            
            # Inject structural modality identity context
            tokens = tokens + self.modality_embeds[i]
            
            # FIX 2: Changed .view() to .reshape() to safely format sliced columns
            p_mask = presence[:, i].reshape(B, 1, 1)
            
            # If present, we keep features and append the present embedding context.
            # If absent, we wipe out normalization/convolutional bias values 
            # leaking from the backbone and replace them with a pure learnable absent embedding.
            tokens = (tokens + self.present_emb[i]) * p_mask + self.absent_emb[i] * (1.0 - p_mask)
            processed_tokens.append(tokens)
            
        # Step A: Global Average Pooling (Spatial to Summary)
        # Stack tokens along modality axis: shape (B, 4, L, C)
        stacked_tokens = torch.stack(processed_tokens, dim=1)
        
        # Pool spatial dimension L: shape (B, 4, C)
        summary_tokens = stacked_tokens.mean(dim=2)
        
        # Step B: Intrinsic Quality Estimator (Q_i)
        q_i = self.iq_mlp(summary_tokens) # Shape: (B, 4, 1)
        
        # Step C: Relational Consistency Estimator (\Delta R_i)
        # 1. Diagonal Masking (Block self-attention)
        attn_mask = torch.zeros(self.num_modalities, self.num_modalities, device=x.device)
        attn_mask.fill_diagonal_(float('-inf'))
        
        # 2. Absence Masking (True = ignore/mask out)
        key_padding_mask = (presence == 0.0) # Shape: (B, 4)
        
        # Execute cross-relational attention
        attn_out, _ = self.rel_attn(
            query=summary_tokens,
            key=summary_tokens,
            value=summary_tokens,
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=False
        )
        
        # Handle N=1 or fully masked rows which natively return NaNs in PyTorch MultiheadAttention
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        
        # Compute \Delta R_i
        delta_r_i = self.rel_mlp(attn_out) # Shape: (B, 4, 1)
        
        # N=1 Handling: If only 1 modality is present, Relational Consistency must default to 0
        presence_count = presence.sum(dim=1, keepdim=True).unsqueeze(-1) # Shape: (B, 1, 1)
        delta_r_i = torch.where(presence_count <= 1, torch.zeros_like(delta_r_i), delta_r_i)
        
        # Step D: Dual-Scale Fusion & Gating
        total_score = q_i + delta_r_i # Shape: (B, 4, 1)
        
        # Masked Softmax: Force missing modalities to -inf so they softmax to 0.0
        total_score = total_score.masked_fill(key_padding_mask.unsqueeze(-1), float('-inf'))
        reliability_weights = torch.softmax(total_score, dim=1) # Shape: (B, 4, 1)
        
        # --- Final Fusion Execution ---
        
        # Apply reliability weights to the original spatial tokens
        # Broadcasting: (B, 4, L, C) * (B, 4, 1, 1) -> shape (B, 4, L, C)
        weighted_tokens = stacked_tokens * reliability_weights.unsqueeze(2)
        
        # Channel-wise Concatenation & Projection 
        # Flatten back into concatenated form: (B, 4, L, C) -> (B, L, 4, C) -> (B, L, 384)
        weighted_concat = weighted_tokens.transpose(1, 2).reshape(B, L, -1)
        
        # Pass through the projection layer to compress and intelligently fuse
        # Output Shape: (B, L, C) = (B, L, 96)
        unified_tensor = self.fusion_proj(weighted_concat)
        
        # Output Shape: (B, L, C)
        return unified_tensor