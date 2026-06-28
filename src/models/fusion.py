# src/models/fusion.py

import torch
import torch.nn as nn

class PresenceAwareCrossModalFusion(nn.Module):
    """Phase 3: Stage 2 - Presence-Aware Cross-Modal Attention Fusion.
    Maps multimodal inputs and adjusts attention weights to isolate and
    compensate for missing modalities using learnable presence tokens.
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

        # Stage 2: Cross-Modal Attention mechanism
        self.cross_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
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
        
        # Automatically detect channel dropouts by tracking voxel intensity sums
        # Resulting shape: (B, num_modalities) where 1 = Present, 0 = Absent
        presence = (x.view(B, self.num_modalities, -1).abs().sum(dim=-1) > 1e-5).float()
        
        processed_tokens = []
        mask_list = []
        
        for i in range(self.num_modalities):
            tokens = modality_tokens[i] # Shape: (B, L, C)
            
            # Inject structural modality identity context
            tokens = tokens + self.modality_embeds[i]
            
            # Extracts the presence status for modality - i, and reshapes it to match the tokens.
            p_mask = presence[:, i].view(B, 1, 1)
            
            # If present, we keep features and append the present embedding context.
            # If absent, we wipe out normalization/convolutional bias values 
            # leaking from the backbone and replace them with a pure learnable absent embedding.
            # Output size: (batch_size, token length, channel(features per token)) = (1, 1728, 96)
            tokens = (tokens + self.present_emb[i]) * p_mask + self.absent_emb[i] * (1.0 - p_mask)
            processed_tokens.append(tokens)
            
            # Construct key padding mask values: True = ignore/mask out during Softmax
            m = (presence[:, i] == 0).unsqueeze(-1).repeat(1, L) # Shape: (B, L)
            mask_list.append(m)
            
        # Concatenate tokens along sequence layout axis: shape (B, 4 * L, C)
        combined_tokens = torch.cat(processed_tokens, dim=1)
        
        # Concatenate padding vectors along key layout axis: shape (B, 4 * L)
        key_padding_mask = torch.cat(mask_list, dim=1)
        
        # Execute Presence-aware multi-head cross attention routing
        attn_out, _ = self.cross_attn(
            query=combined_tokens,
            key=combined_tokens,
            value=combined_tokens,
            key_padding_mask=key_padding_mask
        )
        
        # Apply normalization with residual connection layer
        # Output Shape = (B, L*4, c)
        combined_tokens = self.norm(combined_tokens + attn_out)
        
        # Deconstruct sequence back into 4 separate modality blocks
        # Output Shape = 4 separate (B, L, c)
        split_tokens = torch.chunk(combined_tokens, self.num_modalities, dim=1)
        
        # Enforce clear explicit view restructuring to safely align broadcasting scales
        # Output Shape: (B, 1, 1) = (1, 1, 1); as B==1 in our case
        presence_count = presence.sum(dim=1).clamp(min=1.0).view(B, 1, 1)
        

        # Output Shape: (B, L, C)
        unified_tensor = torch.zeros_like(split_tokens[0])
        for i in range(self.num_modalities):
            p_mask = presence[:, i].view(B, 1, 1)
            unified_tensor = unified_tensor + (split_tokens[i] * p_mask)
            
        # Average only the available modality representations so feature remains consistent when one or more modalities are missing.
        # Output Shape: (B, L, C)
        unified_tensor = unified_tensor / presence_count
        
        # Output Shape: (B, L, C)
        return unified_tensor