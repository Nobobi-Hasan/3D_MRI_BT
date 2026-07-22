# src/models/mamba_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

@torch.jit.script
def accelerated_ssm_loop(x_conv, B_mat, C_mat, dA, dB):
    """Accelerates the sequential state-space scan via a native C++ JIT loop.
    Uses torch.unbind to completely eliminate tensor slicing overhead inside the loop.
    """
    B = x_conv.size(0)
    L = x_conv.size(1)
    D = x_conv.size(2)
    N = B_mat.size(2)
    
    h = torch.zeros(B, D, N, dtype=x_conv.dtype, device=x_conv.device)
    
    # Pre-extract sequences along time dimension to remove indexing overhead
    x_conv_list = torch.unbind(x_conv, dim=1)
    dA_list = torch.unbind(dA, dim=1)
    dB_list = torch.unbind(dB, dim=1)
    C_mat_list = torch.unbind(C_mat, dim=1)
    
    ys = []
    for t in range(L):
        x_t = x_conv_list[t].unsqueeze(-1)          # Shape: (B, D, 1)
        h = dA_list[t] * h + dB_list[t] * x_t       # State Update: (B, D, N)
        C_t = C_mat_list[t].unsqueeze(1)            # Shape: (B, 1, N)
        y = torch.sum(h * C_t, dim=-1)              # Project down: (B, D)
        ys.append(y)
        
    return torch.stack(ys, dim=1)

###############################################################################################



#########################################^^^^^^^^^^^^^^^^^^##############################################
#########################################################################################################

class SingleModalityConvStem3D(nn.Module):
    """Phase 2.1: Hierarchical downsampling local feature stem for a single MRI modality.
    Reduces spatial dimensions from 64x64x64 to 16x16x16 (for preventing token explosion).
    """
    def __init__(self, out_channels=96):
        super().__init__()
        # Layer 1: 64x64x64 -> 32x32x32
        self.layer1 = nn.Sequential(
            nn.Conv3d(1, out_channels // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_channels // 2),
            nn.GELU()
        )
        # Layer 2: 32x32x32 -> 16x16x16
        self.layer2 = nn.Sequential(
            nn.Conv3d(out_channels // 2, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.GroupNorm(8, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        feat1 = self.layer1(x)
        feat2 = self.layer2(feat1)
        return feat1, feat2

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

    def forward(self, x):
        x = self.proj(x)  # Shape: (B, embed_dim, 8, 8, 8)
        B, C, H, W, D = x.shape
        # Flatten spatial structures into a clean 1D token sequence sequence stream
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, H * W * D, C)
        x = self.norm(x)
        return x, (H, W, D)

class BiMambaInnerLayer3D(nn.Module):
    """A highly optimized, hardware-friendly 3D Bidirectional Mamba SSM block."""
    def __init__(self, d_model, d_state=16, d_conv=3, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        
        self.conv1d_fwd = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2, groups=self.d_inner)
        self.conv1d_bwd = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=(d_conv - 1) // 2, groups=self.d_inner)

        self.x_proj_fwd = nn.Linear(self.d_inner, self.d_inner + self.d_state * 2, bias=False)
        self.x_proj_bwd = nn.Linear(self.d_inner, self.d_inner + self.d_state * 2, bias=False)

        self.A_log_fwd = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)))
        self.A_log_bwd = nn.Parameter(torch.log(torch.arange(1, self.d_state + 1).float().repeat(self.d_inner, 1)))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)
        self.dt_init()

    def dt_init(self):
        dt_fwd = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        dt_bwd = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
        self.dt_fwd_param = nn.Parameter(torch.log(dt_fwd))
        self.dt_bwd_param = nn.Parameter(torch.log(dt_bwd))

    # def _ssm_scan(self, u, conv1d_layer, x_proj_layer, dt_param, A_log):
    #     """Streamlined SSM Scan optimized to run efficiently with downsampled sequences."""
    #     x_conv = F.silu(conv1d_layer(u.transpose(1, 2)).transpose(1, 2))
    #     x_pt = x_proj_layer(x_conv)
        
    #     delta, B_mat, C_mat = torch.split(x_pt, [self.d_inner, self.d_state, self.d_state], dim=-1)
    #     delta = F.softplus(delta + dt_param)
        
    #     A = -torch.exp(A_log)
        
    #     dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
    #     dB = delta.unsqueeze(-1) * B_mat.unsqueeze(-2)
        
    #     # Sequentially scan tokens; sequence length (1,728) prevents performance bottlenecks
    #     h = torch.zeros(u.size(0), self.d_inner, self.d_state, device=u.device)
    #     ys = []
    #     for t in range(u.size(1)):
    #         x_t = x_conv[:, t].unsqueeze(-1)
    #         h = dA[:, t] * h + dB[:, t] * x_t
    #         C_t = C_mat[:, t].unsqueeze(1)
    #         y = torch.sum(h * C_t, dim=-1)
    #         ys.append(y)
            
    #     return torch.stack(ys, dim=1)

    def _ssm_scan(self, u, conv1d_layer, x_proj_layer, dt_param, A_log):
        """Pre-computes state matrices and transfers processing to the JIT compiled loop."""
        x_conv = F.silu(conv1d_layer(u.transpose(1, 2)).transpose(1, 2))
        x_pt = x_proj_layer(x_conv)
        
        delta, B_mat, C_mat = torch.split(x_pt, [self.d_inner, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(delta + dt_param)
        
        A = -torch.exp(A_log)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dB = delta.unsqueeze(-1) * B_mat.unsqueeze(-2)
        
        # Offload sequence scan to compiled C++ runtime helper
        return accelerated_ssm_loop(x_conv, B_mat, C_mat, dA, dB)


    def forward(self, x):
        projected = self.in_proj(x)
        u, residual = torch.chunk(projected, 2, dim=-1)
        
        out_fwd = self._ssm_scan(u, self.conv1d_fwd, self.x_proj_fwd, self.dt_fwd_param, self.A_log_fwd)
        
        u_bwd = torch.flip(u, dims=[1])
        out_bwd = self._ssm_scan(u_bwd, self.conv1d_bwd, self.x_proj_bwd, self.dt_bwd_param, self.A_log_bwd)
        out_bwd = torch.flip(out_bwd, dims=[1])
        
        fused = (out_fwd + out_bwd) * F.silu(residual)
        return self.out_proj(fused)

class BiMambaEncoder3D(nn.Module):
    """Residual wrapper block grouping sequential bidirectional Mamba layers."""
    def __init__(self, d_model=96, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                nn.LayerNorm(d_model),
                BiMambaInnerLayer3D(d_model=d_model)
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for norm, mamba in self.layers:
            x = x + mamba(norm(x))
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
        
        for i in range(num_mods):
            mod_channel = x[:, i:i+1, :, :, :]
            feat1, feat2 = self.stems[i](mod_channel)
            tokens, patch_shape = self.patch_embeds[i](feat2)
            
            if i == 0:
                spatial_shapes = patch_shape  # Expected shape layout: (8, 8, 8)
                
            encoded_tokens = self.modality_encoders[i](tokens)
            modality_tokens.append(encoded_tokens)
            
            # Append spatial maps for cross-modal skip fusion layout
            feat1_list.append(feat1)
            feat2_list.append(feat2)
            
        # Concatenate multi-modal spatial maps along the channel axis to preserve structural information
        skip_features = [
            torch.cat(feat1_list, dim=1),  # Combined Level 1 maps: shape (B, 4 * 48, 32, 32, 32)
            torch.cat(feat2_list, dim=1)   # Combined Level 2 maps: shape (B, 4 * 96, 16, 16, 16)
        ]
            
        return modality_tokens, spatial_shapes, skip_features

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
                BiMambaInnerLayer3D(d_model=embed_dim)
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
        for norm, mamba in self.layers:
            x = x + mamba(norm(x))
            
        return self.final_norm(x)