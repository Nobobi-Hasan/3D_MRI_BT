# src/models/classification.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class RegionAttentionAggregator(nn.Module):
    """Phase 5.3: Region Attention Aggregator.
    Weights WT, TC, and ET embeddings based on their clinical relevance, 
    provides both - better feature mix and model explainability.
    """
    def __init__(self, embed_dim=96):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, region_embeddings):
        """
        Args:
            region_embeddings (Tensor): Stacked embeddings of shape (B, 3, embed_dim)
        Returns:
            fused_representation (Tensor): Aggregated vector of shape (B, embed_dim)
            weights (Tensor): Raw attention weights of shape (B, 3, 1) for interpretability
        """
        # Compute scalar importance scores for each region embedding
        scores = self.attn_net(region_embeddings)  # Shape: (B, 3, 1)
        
        # Normalize scores across the 3 regions using Softmax
        weights = F.softmax(scores, dim=1)  # Shape: (B, 3, 1)
        
        # Compute the attention-weighted sum of the embeddings
        fused_representation = torch.sum(region_embeddings * weights, dim=1)  # Shape: (B, embed_dim)
        
        return fused_representation, weights

class MorphologyGuidedClassifier(nn.Module):
    """Phase 5: Morphology-Guided Brain Tumor Grading Classifier.
    Extracts regional multi-modal characteristics using segmentation parts, 
    aggregates features via an attention network, and classifies samples into HGG or LGG.
    Use learnable presence/absence tokens to handle missing tumor sub-regions (like missing ET).
    """
    def __init__(self, embed_dim=96, num_classes=2):
        super().__init__()
        self.embed_dim = embed_dim
        
        # Phase 5.1 & 5.2: Learnable regional presence/absence tokens (0: WT, 1: TC, 2: ET)
        self.present_region_embeds = nn.Parameter(torch.randn(3, embed_dim) * 0.02) # Shape: (3, embed_dim)
        self.absent_region_embeds = nn.Parameter(torch.randn(3, embed_dim) * 0.02) # Shape: (3, embed_dim)

        # 5.3 Region Attention Aggregator Module
        self.aggregator = RegionAttentionAggregator(embed_dim=embed_dim)
        
        # 5.4 Final MLP Classification Head
        self.mlp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )

    def _masked_feature_pooling(self, features, mask):
        """Phase 5.1: Averages features isolated inside a binary mask."""
        masked_features = features * mask
        spatial_sum = masked_features.sum(dim=(2, 3, 4))  # Shape: (B, C)
        voxel_count = mask.sum(dim=(2, 3, 4)).clamp(min=1.0)  # Shape: (1, 1)
        return spatial_sum / voxel_count # Shape: (B, C)

    def forward(self, seg_logits, refined_features, gt_labels=None):
        """
        Args:
            seg_logits (Tensor): Segmentation predictions. Shape: (B, 4, 96, 96, 96)
            refined_features (Tensor): Spatially attended volume from decoder. Shape: (B, embed_dim, 12, 12, 12)
            gt_labels (Tensor, optional): Ground-truth segmentation map used during training. Shape: (B, 1, 96, 96, 96)
        Returns:
            cls_logits (Tensor): Classification grading logits. Shape: (B, 2)
            attn_weights (Tensor): Region importance weights. Shape: (B, 3, 1)
        """
        B, C, H, W, D = refined_features.shape
        
        # Phase 5.1: Determine whether to use Ground-Truth or Predicted masks.
        if self.training and gt_labels is not None:
            seg_map = gt_labels  # Shape: (B, 1, 96, 96, 96)
        else:
            seg_map = torch.argmax(seg_logits, dim=1, keepdim=True)  # Shape: (B, 1, 96, 96, 96)
            
        # Extract the 3 tumor sub-regions using BraTS segment labels
        wt_mask = (seg_map == 1) | (seg_map == 2) | (seg_map == 3)  # Whole Tumor
        tc_mask = (seg_map == 1) | (seg_map == 3)                   # Tumor Core
        et_mask = (seg_map == 3)                                    # Enhancing Tumor
        
        # Downsample masks to 12x12x12 to align with the latent feature dimensions
        wt_mask = F.interpolate(wt_mask.float(), size=(H, W, D), mode="nearest")
        tc_mask = F.interpolate(tc_mask.float(), size=(H, W, D), mode="nearest")
        et_mask = F.interpolate(et_mask.float(), size=(H, W, D), mode="nearest")
        
        # Track regional anatomical presence states per sample across the batch
        wt_present = (wt_mask.sum(dim=(2, 3, 4)) > 1e-5).float().view(B, 1)
        tc_present = (tc_mask.sum(dim=(2, 3, 4)) > 1e-5).float().view(B, 1)
        et_present = (et_mask.sum(dim=(2, 3, 4)) > 1e-5).float().view(B, 1)

        # Execute Morphology-Guided Feature Pooling (MGFP)
        wt_emb_raw = self._masked_feature_pooling(refined_features, wt_mask)  # Shape: (B, embed_dim)
        tc_emb_raw = self._masked_feature_pooling(refined_features, tc_mask)  # Shape: (B, embed_dim)
        et_emb_raw = self._masked_feature_pooling(refined_features, et_mask)  # Shape: (B, embed_dim)
        
        # Inject learnable tokens to accurately represent present features or absent features
        wt_emb = (wt_emb_raw + self.present_region_embeds[0]) * wt_present + self.absent_region_embeds[0] * (1.0 - wt_present)
        tc_emb = (tc_emb_raw + self.present_region_embeds[1]) * tc_present + self.absent_region_embeds[1] * (1.0 - tc_present)
        et_emb = (et_emb_raw + self.present_region_embeds[2]) * et_present + self.absent_region_embeds[2] * (1.0 - et_present)

        # Phase 5.2: Stack independent vectors into a unified tensor layout
        region_embeddings = torch.stack([wt_emb, tc_emb, et_emb], dim=1)  # Shape: (B, 3, embed_dim)
        
        # Phase 5.3: Prioritize regional features via Cross-Region Attention
        fused_features, attn_weights = self.aggregator(region_embeddings)  # Fused: (B, embed_dim)
        
        # Phase 5.4: Generate grading diagnosis logits (LGG vs HGG)
        cls_logits = self.mlp_head(fused_features)  # Shape: (B, 2)
        
        return cls_logits, attn_weights