# src/models/decoder.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention3D(nn.Module):
    """Phase 4.1: Lightweight 3D Spatial Attention Refinement module.
    Emphasize tumor regions and refine boundary localization.
    """
    def __init__(self, channels):
        super().__init__()
        self.spatial_conv = nn.Conv3d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Generate spatial attention weights: Shape (B, 1, H, W, D)
        attention_weights = self.sigmoid(self.spatial_conv(x))
        return x * attention_weights

class DecoderBlock3D(nn.Module):
    """3D upsampling and feature convolution."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.conv(self.upsample(x))

class SegmentationDecoder3D(nn.Module):
    """Phase 4.2: 3D Segmentation Decoder.
    Reconstructs volumetric representations back to the original input resolution
    and outputs 4 channels as BraTS labels:
    (0: Background, 1: NCR, 2: ED, 3: ET) to match multi-class Softmax losses.
    """
    def __init__(self, embed_dim=96, out_channels=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 4.1 Spatial Attention block
        self.spatial_attention = SpatialAttention3D(channels=embed_dim)
        
        # 4.2 Hierarchical Upsampling Blocks: Reconstructs 12x12x12 -> 96x96x96
        self.layer1 = DecoderBlock3D(in_channels=embed_dim, out_channels=64)  # 12x12x12 -> 24x24x24
        self.layer2 = DecoderBlock3D(in_channels=64, out_channels=32)        # 24x24x24 -> 48x48x48
        self.layer3 = DecoderBlock3D(in_channels=32, out_channels=16)        # 48x48x48 -> 96x96x96
        
        # Final Segmentation Head mapping to 4 target structural target regions
        self.seg_head = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, latent_tokens, spatial_shape):
        """
        Args:
            latent_tokens (Tensor): Shared Unified Latent Representation from the backbone.
                                    Shape: (B, L, embed_dim) where L = H_p * W_p * D_p (1728)
            spatial_shape (tuple): Target grid shape (H_p, W_p, D_p) matching token shape (12, 12, 12)
        Returns:
            logits (Tensor): 3D segmentation logits of shape (B, 4, 96, 96, 96)
            refined_features (Tensor): to assist Morphology-Guided Classification.
        """
        B, L, C = latent_tokens.shape
        H_p, W_p, D_p = spatial_shape
        
        # Re-assemble 1D token streams back into standard 3D spatial grids
        x = latent_tokens.view(B, H_p, W_p, D_p, C).permute(0, 4, 1, 2, 3).contiguous() # (B, embed_dim, 12, 12, 12)
        
        # Phase 4.1: Apply spatial attention refinement
        refined_features = self.spatial_attention(x)
        
        # Phase 4.2: Progressive volumetric reconstruction path
        out = self.layer1(refined_features) # (B, 64, 24, 24, 24)
        out = self.layer2(out)             # (B, 32, 48, 48, 48)
        out = self.layer3(out)             # (B, 16, 96, 96, 96)
        
        logits = self.seg_head(out)        # (B, 4, 96, 96, 96)
        
        return logits, refined_features