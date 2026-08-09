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
        
        # 4.2 Hierarchical Upsampling Blocks: Reconstructs 8x8x8 -> 64x64x64
        # Layer 1 upsamples the bottleneck: 8x8x8 -> 16x16x16
        self.layer1 = DecoderBlock3D(in_channels=embed_dim, out_channels=64)  
        
        # 1x1x1 Conv projections to compress multi-modal skip features and save memory
        self.skip3_proj = nn.Sequential(
            nn.Conv3d(384, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128),
            nn.GELU()
        )
        self.skip2_proj = nn.Sequential(
            nn.Conv3d(192, 64, kernel_size=1, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU()
        )
        self.skip1_proj = nn.Sequential(
            nn.Conv3d(96, 32, kernel_size=1, bias=False),
            nn.GroupNorm(8, 32),
            nn.GELU()
        )
        
        # Layer 2 handles Layer 1 output (64 ch) + Compressed Level 3 skips (128 ch) -> Total 192 ch
        # 16x16x16 -> 32x32x32
        self.layer2 = DecoderBlock3D(in_channels=64 + 128, out_channels=32)        
        
        # Layer 3 handles Layer 2 output (32 ch) + Compressed Level 2 skips (64 ch) -> Total 96 ch
        # 32x32x32 -> 64x64x64
        self.layer3 = DecoderBlock3D(in_channels=32 + 64, out_channels=16)        
        
        # Layer 4 handles Layer 3 output (16 ch) + Compressed Level 1 skips (32 ch) -> Total 48 ch
        # 64x64x64 -> 64x64x64 (Mixing Block, no upsampling)
        self.layer4 = nn.Sequential(
            nn.Conv3d(16 + 32, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.GELU()
        )
        
        # Final Segmentation Head mapping to 4 target structural target regions
        self.seg_head = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, latent_tokens, spatial_shape, skip_features):
        """
        Args:
            latent_tokens (Tensor): Shared Unified Latent Representation from the backbone.
                                    Shape: (B, L, embed_dim) where L = H_p * W_p * D_p (512)
            spatial_shape (tuple): Target grid shape (H_p, W_p, D_p) matching token shape (8, 8, 8)
            skip_features (list of 3 Tensors): Multi-scale spatial feature maps from the encoder stems.
                                               skip_features[0]: shape (B, 96, 64, 64, 64)
                                               skip_features[1]: shape (B, 192, 32, 32, 32)
                                               skip_features[2]: shape (B, 384, 16, 16, 16)
        Returns:
            logits (Tensor): 3D segmentation logits of shape (B, 4, 64, 64, 64)
        """
        B, L, C = latent_tokens.shape
        H_p, W_p, D_p = spatial_shape
        
        # Re-assemble 1D token streams back into standard 3D spatial grids
        x = latent_tokens.view(B, H_p, W_p, D_p, C).permute(0, 4, 1, 2, 3).contiguous() # (B, embed_dim, 8, 8, 8)
        
        # Phase 4.1: Apply spatial attention refinement
        refined_features = self.spatial_attention(x)
        
        # Phase 4.2: Progressive volumetric reconstruction path with skip connections
        out = self.layer1(refined_features) # (B, 64, 16, 16, 16)
        
        # Project and compress Level 3 multi-scale spatial maps to mitigate modality dropout scale variance
        skip3 = self.skip3_proj(skip_features[2]) # (B, 128, 16, 16, 16)
        
        # Concatenate with Level 3 multi-scale spatial maps along the channels axis
        out = torch.cat([out, skip3], dim=1) # (B, 64 + 128, 16, 16, 16)
        out = self.layer2(out)             # (B, 32, 32, 32, 32)
        
        # Project and compress Level 2 multi-scale spatial maps to mitigate modality dropout scale variance
        skip2 = self.skip2_proj(skip_features[1]) # (B, 64, 32, 32, 32)
        
        # Concatenate with Level 2 multi-scale spatial maps along the channels axis
        out = torch.cat([out, skip2], dim=1) # (B, 32 + 64, 32, 32, 32)
        out = self.layer3(out)             # (B, 16, 64, 64, 64)
        
        # Project and compress Level 1 multi-scale spatial maps (High-res edges)
        skip1 = self.skip1_proj(skip_features[0]) # (B, 32, 64, 64, 64)
        
        # Concatenate with Level 1 multi-scale spatial maps along the channels axis
        out = torch.cat([out, skip1], dim=1) # (B, 16 + 32, 64, 64, 64)
        out = self.layer4(out)             # (B, 16, 64, 64, 64)
        
        logits = self.seg_head(out)        # (B, 4, 64, 64, 64)
        
        return logits


class AuxiliaryDecoder3D(nn.Module):
    """Phase 4.3: Shared-Weight Auxiliary Decoder.
    Reconstructs volumetric representations for single unmasked modalities
    to provide independent supervision and ensure balanced modality learning.
    """
    def __init__(self, embed_dim=96, out_channels=4):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 4.3.1 Spatial Attention block (Shared architecture logic)
        self.spatial_attention = SpatialAttention3D(channels=embed_dim)
        
        # 4.3.2 Hierarchical Upsampling Blocks: Reconstructs 8x8x8 -> 64x64x64
        self.layer1 = DecoderBlock3D(in_channels=embed_dim, out_channels=64)  
        
        # Layer 2 handles Layer 1 output (64 ch) + Raw Level 3 single-modal skips (96 ch) -> Total 160 ch
        self.layer2 = DecoderBlock3D(in_channels=64 + 96, out_channels=32)        
        
        # Layer 3 handles Layer 2 output (32 ch) + Raw Level 2 single-modal skips (48 ch) -> Total 80 ch
        self.layer3 = DecoderBlock3D(in_channels=32 + 48, out_channels=16)        
        
        # Layer 4 handles Layer 3 output (16 ch) + Raw Level 1 single-modal skips (24 ch) -> Total 40 ch
        # Mixing Block (Stride-1, no upsampling)
        self.layer4 = nn.Sequential(
            nn.Conv3d(16 + 24, 16, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, 16),
            nn.GELU()
        )
        
        # Final Segmentation Head mapping to 4 target structural target regions
        self.seg_head = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, latent_tokens, spatial_shape, skip_features):
        """
        Args:
            latent_tokens (Tensor): Single modality latent representation.
                                    Shape: (B, L, embed_dim)
            spatial_shape (tuple): Target grid shape (H_p, W_p, D_p) matching token shape (8, 8, 8)
            skip_features (list of 3 Tensors): Multi-scale spatial feature maps from ONE encoder stem.
                                               skip_features[0]: shape (B, 24, 64, 64, 64)
                                               skip_features[1]: shape (B, 48, 32, 32, 32)
                                               skip_features[2]: shape (B, 96, 16, 16, 16)
        Returns:
            logits (Tensor): 3D segmentation logits of shape (B, 4, 64, 64, 64)
        """
        B, L, C = latent_tokens.shape
        H_p, W_p, D_p = spatial_shape
        
        # Re-assemble 1D token streams back into standard 3D spatial grids
        x = latent_tokens.view(B, H_p, W_p, D_p, C).permute(0, 4, 1, 2, 3).contiguous()
        
        # Phase 4.3.1: Apply spatial attention refinement
        refined_features = self.spatial_attention(x)
        
        # Phase 4.3.2: Progressive volumetric reconstruction path with skip connections
        out = self.layer1(refined_features) 
        
        # Direct concatenation of single-modality Level 3 skips (no projection)
        out = torch.cat([out, skip_features[2]], dim=1) 
        out = self.layer2(out)             
        
        # Direct concatenation of single-modality Level 2 skips (no projection)
        out = torch.cat([out, skip_features[1]], dim=1) 
        out = self.layer3(out)             
        
        # Direct concatenation of single-modality Level 1 skips (no projection)
        out = torch.cat([out, skip_features[0]], dim=1) 
        out = self.layer4(out)             
        
        logits = self.seg_head(out)        
        
        return logits