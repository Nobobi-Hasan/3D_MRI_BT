import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from mamba_ssm import Mamba

class SingleModalityConvStem3D(nn.Module):
    """Phase 2.1: Hierarchical downsampling local feature stem for a single MRI modality.
    Reduces spatial dimensions from 64x64x64 to 16x16x16 (for preventing token explosion).
    """
    def __init__(self, out_channels=96):
        super().__init__()
        # Layer 1: 64x64x64 -> 64x64x64 (New Stride-1 block for high-res edge details)
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, out_channels // 4, kernel_size=3, stride=1, padding=1, bias=False),
            nn.GroupNorm(8, out_channels // 4),
            nn.GELU()
        )
        # Layer 2: 64x64x64 -> 32x32x32 (Former Layer 1)
        self.layer2 = nn.Sequential(
            nn.Conv3d(out_channels // 4, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_channels // 2),
            nn.GELU()
        )
        # Layer 3: 32x32x32 -> 16x16x16 (Former Layer 2)
        self.layer3 = nn.Sequential(
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        return feat1, feat2, feat3

class OverlappingPatchEmbed3D(nn.Module):
    """Phase 2.2: Converts downsampled 3D feature (from 2.1) maps into overlapping volumetric tokens.
    Transforms 16x16x16 grid into 8x8x8 = 512 tokens.
    """
    def __init__(self, patch_size=4, stride=2, in_channels=96, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv3d(
            in_channels, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=stride, 
            padding=(patch_size - stride) // 2
        )
        self.norm = nn.LayerNorm(embed_dim)
        
        # --- Learnable 3D Absolute Positional Embeddings ---
        # 8x8x8 spatial grid = 512 tokens
        self.pos_embed = nn.Parameter(torch.randn(1, 512, embed_dim) * 0.02)
        # --------------------------------------------------------

    def forward(self, x):
        x = self.proj(x)  # Shape: (B, embed_dim, 8, 8, 8)
        B, C, H, W, D = x.shape
        # Flatten spatial structures into a clean 1D token sequence sequence stream
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, H * W * D, C)
        x = self.norm(x)
        
        # --- Inject 3D positional awareness directly into the tokens ---
        x = x + self.pos_embed
        # --------------------------------------------------------------------
        
        return x, (H, W, D)

class BiMambaInnerLayer3D(nn.Module):
    """A highly optimized, hardware-friendly 3D Bidirectional Mamba SSM block."""
    def __init__(self, d_model=96, d_state=16, d_conv=3, expand=2):
        super().__init__()
        # The official mamba-ssm package from PyPI only does forward causal scans natively.
        # To recreate the Vision Mamba (Vim) "v2" bidirectional behavior, we use two separate 
        # hardware-fused blocks (one for the forward pass, one for the backward pass).
        
        self.mamba_fwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        self.mamba_bwd = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x):
        # Forward sequence scan (Hardware fused)
        out_fwd = self.mamba_fwd(x)
        
        # Backward sequence scan (Flip sequence length dimension L -> B, L, C)
        x_flipped = torch.flip(x, dims=[1])
        out_bwd = self.mamba_bwd(x_flipped)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        # Fuse the bidirectional features
        return out_fwd + out_bwd

class BiMambaEncoder3D(nn.Module):
    """Residual wrapper block grouping sequential bidirectional Mamba layers."""
    def __init__(self, d_model=96, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(d_model),
                BiMambaInnerLayer3D(d_model=d_model),
                nn.LayerNorm(d_model),
                nn.Sequential(
                    nn.Linear(d_model, d_model * 2),
                    nn.GELU(),
                    nn.Linear(d_model * 2, d_model)
                )
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for norm1, mamba, norm2, mlp in self.layers:
            x = x + mamba(norm1(x))
            x = x + mlp(norm2(x))
        return x

class MambaBackbone(nn.Module):
    """Main Feature Extraction Backbone - Phase 2 and Phase 3 (Stage 1)."""
    def __init__(self, num_modalities=4, embed_dim=96, mamba_depth=2):
        super().__init__()
        self.stems = nn.ModuleList([
            SingleModalityConvStem3D(out_channels=embed_dim) for _ in range(num_modalities)
        ])
        self.patch_embeds = nn.ModuleList([
            OverlappingPatchEmbed3D(patch_size=4, stride=2, in_channels=embed_dim, embed_dim=embed_dim)
            for _ in range(num_modalities)
        ])
        self.modality_encoders = nn.ModuleList([
            BiMambaEncoder3D(d_model=embed_dim, depth=mamba_depth) for _ in range(num_modalities)
        ])

    def forward(self, x):
        B, num_mods, H, W, D = x.shape
        modality_tokens = []
        spatial_shapes = []
        
        # Lists to collect high-resolution multi-scale spatial feature maps across modalities
        feat1_list = []
        feat2_list = []
        feat3_list = []
        
        for i in range(num_mods):
            mod_channel = x[:, i:i+1, :, :, :]
            feat1, feat2, feat3 = self.stems[i](mod_channel)
            tokens, patch_shape = self.patch_embeds[i](feat3)
            
            if i == 0:
                spatial_shapes = patch_shape  # Expected shape layout: (8, 8, 8)
                
            encoded_tokens = self.modality_encoders[i](tokens)
            modality_tokens.append(encoded_tokens)
            
            # Append spatial maps for cross-modal skip fusion layout
            feat1_list.append(feat1)
            feat2_list.append(feat2)
            feat3_list.append(feat3)
            
        # Concatenate multi-modal spatial maps along the channel axis to preserve structural information
        skip_features = [
            torch.cat(feat1_list, dim=1),  # Combined Level 1 maps: shape (B, 4 * 24, 64, 64, 64)
            torch.cat(feat2_list, dim=1),  # Combined Level 2 maps: shape (B, 4 * 48, 32, 32, 32)
            torch.cat(feat3_list, dim=1)   # Combined Level 3 maps: shape (B, 4 * 96, 16, 16, 16)
        ]
        
        # Pack individual multi-scale spatial maps for auxiliary single-modality decoding
        single_skip_features = [feat1_list, feat2_list, feat3_list]
            
        return modality_tokens, spatial_shapes, skip_features, single_skip_features

class SharedDeepMambaBackbone(nn.Module):
    """Phase 3: Stage 3 - Shared Deep Mamba Backbone.
    Processes the unified multi-modal feature tensor to model global tumor semantics
    and whole-brain contextual relationships across a deeper network block.
    """
    def __init__(self, embed_dim=96, mamba_depth=4):
        super().__init__()
        # Stacks deep bidirectional SSM layers to build structural amodal context
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(embed_dim),
                BiMambaInnerLayer3D(d_model=embed_dim),
                nn.LayerNorm(embed_dim),
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim * 2),
                    nn.GELU(),
                    nn.Linear(embed_dim * 2, embed_dim)
                )
            ]) for _ in range(mamba_depth)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        """
        Args:
            x (Tensor): Unified Multi-Modal Feature Tensor of shape (B, L, C)
        Returns:
            Tensor: Shared Unified Latent Representation of shape (B, L, C)
        """
        # Execute sequential residual forwarding loops through the deep backbone
        for norm1, mamba, norm2, mlp in self.layers:
            x = x + mamba(norm1(x))
            x = x + mlp(norm2(x))
            
        return self.final_norm(x)