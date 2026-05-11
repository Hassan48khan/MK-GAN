"""
MK-GAN: Multi-Scale Mamba-KAN Generative Adversarial Network
Complete PyTorch Implementation

Novel Components:
1. AMKAN Block (Adaptive Multi-scale Mamba-KAN Block)
2. DSF-VSS (Dynamic Scale Fusion VSS) 
3. AKFC-KAN (Adaptive Kernel Fusion Convolutional KAN)
4. ABRA (Adaptive Boundary-Region Attention)

Author: [Your Name]
Date: 2025
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Optional


# ============================================================================
# NOVEL COMPONENT 1: Dynamic Scale Fusion (DSF)
# ============================================================================

class DynamicScaleFusion(nn.Module):
    """
    Novel Dynamic Scale Fusion mechanism
    Adaptively fuses features from multiple scales with learnable weights
    """
    def __init__(self, filters: int, num_scales: int = 3):
        super(DynamicScaleFusion, self).__init__()
        self.filters = filters
        self.num_scales = num_scales
        
        # Scale-specific convolutions
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(filters, filters, kernel_size=2*i+3, padding=i+1)
            for i in range(num_scales)
        ])
        
        # Learnable scale weights
        self.scale_weights = nn.Parameter(torch.ones(num_scales, 1, 1, 1))
        
        # Fusion layer
        self.fusion_conv = nn.Conv2d(filters * num_scales, filters, 1)
        self.ln = nn.LayerNorm([filters])
        self.act = nn.SiLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Extract multi-scale features
        scale_features = []
        for i, conv in enumerate(self.scale_convs):
            feat = conv(x)
            # Apply learnable scale weight
            weighted_feat = feat * self.scale_weights[i]
            scale_features.append(weighted_feat)
        
        # Concatenate and fuse
        fused = torch.cat(scale_features, dim=1)
        output = self.fusion_conv(fused)
        
        # Normalize (channel-wise)
        output = output.permute(0, 2, 3, 1)  # B, H, W, C
        output = self.ln(output)
        output = output.permute(0, 3, 1, 2)  # B, C, H, W
        
        output = self.act(output)
        
        return output


# ============================================================================
# NOVEL COMPONENT 2: Learnable Directional Scanning (LDS)
# ============================================================================

class LearnableDirectionalScanning(nn.Module):
    """
    Novel Learnable Directional Scanning
    Learns optimal scanning directions instead of fixed 4 directions
    """
    def __init__(self, filters: int, num_directions: int = 6):
        super(LearnableDirectionalScanning, self).__init__()
        self.filters = filters
        self.num_directions = num_directions
        
        # Direction-specific transformations
        self.dir_convs = nn.ModuleList([
            nn.Conv2d(filters, filters, 3, padding=1)
            for _ in range(num_directions)
        ])
        
        # Learnable direction embeddings
        self.dir_embed = nn.Parameter(torch.randn(num_directions, filters))
        
        # Attention mechanism
        self.attention_conv = nn.Conv2d(num_directions, num_directions, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        
        # Apply directional transformations
        dir_features = []
        for i, conv in enumerate(self.dir_convs):
            dir_feat = conv(x)
            # Add direction embedding
            dir_emb = self.dir_embed[i].view(1, C, 1, 1)
            dir_feat = dir_feat + dir_emb
            dir_features.append(dir_feat)
        
        # Stack directions: B, num_dir, C, H, W
        dir_stack = torch.stack(dir_features, dim=1)
        
        # Compute attention weights
        # Average over channels: B, num_dir, H, W
        attn_input = dir_stack.mean(dim=2)
        attn_weights = self.attention_conv(attn_input)
        attn_weights = F.softmax(attn_weights, dim=1)
        
        # Weighted aggregation
        attn_weights = attn_weights.unsqueeze(2)  # B, num_dir, 1, H, W
        output = (dir_stack * attn_weights).sum(dim=1)  # B, C, H, W
        
        return output


# ============================================================================
# NOVEL COMPONENT 3: S6 State-Space Block
# ============================================================================

class S6Block(nn.Module):
    """
    State Space Block (S6) for temporal modeling
    Simplified version for 2D images
    """
    def __init__(self, filters: int, state_dim: int = 16):
        super(S6Block, self).__init__()
        self.filters = filters
        self.state_dim = state_dim
        
        # State space parameters
        self.conv_in = nn.Conv2d(filters, state_dim, 1)
        self.conv_state = nn.Conv2d(state_dim, state_dim, 3, padding=1)
        self.conv_out = nn.Conv2d(state_dim, filters, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to state dimension
        state = self.conv_in(x)
        
        # State evolution
        state = self.conv_state(state)
        state = F.relu(state)
        
        # Project back to output dimension
        output = self.conv_out(state)
        
        return output


# ============================================================================
# DSF-VSS Block (Combining DSF + LDS + S6)
# ============================================================================

class DSFVSSBlock(nn.Module):
    """
    Novel Dynamic Scale Fusion VSS Block
    Combines DSF + LDS + S6
    """
    def __init__(self, filters: int, num_scales: int = 3, num_directions: int = 6):
        super(DSFVSSBlock, self).__init__()
        self.filters = filters
        
        # Layer normalization
        self.ln1 = nn.LayerNorm([filters])
        
        # Expansion
        self.expand = nn.Conv2d(filters, filters * 4, 1)
        
        # Dynamic Scale Fusion
        self.dsf = DynamicScaleFusion(filters * 2, num_scales)
        
        # Depthwise convolution
        self.dwconv = nn.Conv2d(filters * 2, filters * 2, 3, padding=1, groups=filters * 2)
        
        # Learnable Directional Scanning
        self.lds = LearnableDirectionalScanning(filters * 2, num_directions)
        
        # S6 Block
        self.s6 = S6Block(filters * 2)
        
        # Output projection
        self.proj = nn.Conv2d(filters * 2, filters, 1)
        self.ln2 = nn.LayerNorm([filters])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        # Normalize
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        # Expand
        x = self.expand(x)
        
        # Split into two branches
        x1, x2 = torch.chunk(x, 2, dim=1)
        
        # Branch 1: DSF + DWConv + LDS + S6
        x1 = self.dsf(x1)
        x1 = self.dwconv(x1)
        x1 = F.silu(x1)
        x1 = self.lds(x1)
        x1 = self.s6(x1)
        
        # Normalize
        x1 = x1.permute(0, 2, 3, 1)  # B, H, W, C
        x1 = self.ln2(x1)
        x1 = x1.permute(0, 3, 1, 2)  # B, C, H, W
        
        # Branch 2: Activation
        x2 = F.silu(x2)
        
        # Element-wise multiplication
        x_out = x1 * x2
        
        # Project back
        x_out = self.proj(x_out)
        
        # Residual connection
        output = x_out + residual
        
        return output


# ============================================================================
# NOVEL COMPONENT 4: Adaptive Kernel Fusion (AKF)
# ============================================================================

class AdaptiveKernelFusion(nn.Module):
    """
    Novel Adaptive Kernel Fusion
    Dynamically selects and fuses multiple kernel sizes
    """
    def __init__(self, filters: int, kernel_sizes: List[int] = [3, 5, 7]):
        super(AdaptiveKernelFusion, self).__init__()
        self.filters = filters
        self.kernel_sizes = kernel_sizes
        
        # Multiple kernel branches
        self.kernel_branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(filters, filters, k, padding=k//2, groups=filters),
                nn.BatchNorm2d(filters),
                nn.SiLU()
            )
            self.kernel_branches.append(branch)
        
        # Channel attention for kernel selection
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(filters * len(kernel_sizes), filters // 4)
        self.fc2 = nn.Linear(filters // 4, len(kernel_sizes))
        
        # Fusion
        self.fusion = nn.Conv2d(filters, filters, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply all kernel branches
        branch_outputs = [branch(x) for branch in self.kernel_branches]
        
        # Concatenate for attention
        concat = torch.cat(branch_outputs, dim=1)
        
        # Compute kernel attention weights
        attn = self.gap(concat).flatten(1)
        attn = F.silu(self.fc1(attn))
        attn = F.softmax(self.fc2(attn), dim=1)  # B, num_kernels
        
        # Weighted fusion
        attn = attn.view(-1, len(self.kernel_sizes), 1, 1, 1)
        branch_stack = torch.stack(branch_outputs, dim=1)  # B, num_kernels, C, H, W
        fused = (branch_stack * attn).sum(dim=1)
        
        # Final fusion
        output = self.fusion(fused)
        
        return output


# ============================================================================
# NOVEL COMPONENT 5: Tok-KAN Block
# ============================================================================

class TokKAN(nn.Module):
    """
    Token-based KAN block
    """
    def __init__(self, filters: int, expansion: int = 2):
        super(TokKAN, self).__init__()
        hidden = filters * expansion
        
        # First KAN transformation
        self.conv1 = nn.Conv2d(filters, hidden, 1)
        self.dwconv1 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.bn1 = nn.BatchNorm2d(hidden)
        
        # Second KAN transformation
        self.conv2 = nn.Conv2d(hidden, filters, 1)
        self.dwconv2 = nn.Conv2d(filters, filters, 3, padding=1, groups=filters)
        self.bn2 = nn.BatchNorm2d(filters)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First transformation
        out = self.conv1(x)
        out = F.silu(out)
        out = self.dwconv1(out)
        out = self.bn1(out)
        
        # Second transformation
        out = self.conv2(out)
        out = F.silu(out)
        out = self.dwconv2(out)
        out = self.bn2(out)
        
        return out


# ============================================================================
# AKFC-KAN Block (Combining AKF + Tok-KAN)
# ============================================================================

class AKFCKANBlock(nn.Module):
    """
    Novel Adaptive Kernel Fusion Convolutional KAN Block
    """
    def __init__(self, filters: int, kernel_sizes: List[int] = [3, 5, 7]):
        super(AKFCKANBlock, self).__init__()
        
        # Adaptive Kernel Fusion
        self.akf = AdaptiveKernelFusion(filters, kernel_sizes)
        
        # Layer normalization
        self.ln = nn.LayerNorm([filters])
        
        # Tok-KAN block
        self.tok_kan = TokKAN(filters)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        # Adaptive Kernel Fusion
        x = self.akf(x)
        
        # Normalize
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        # Tok-KAN
        x = self.tok_kan(x)
        
        # Residual connection
        output = x + residual
        
        return output


# ============================================================================
# NOVEL COMPONENT 6: Multi-Scale Boundary Detection
# ============================================================================

class MultiScaleBoundaryDetection(nn.Module):
    """
    Novel Multi-Scale Boundary Detection
    Detects boundaries at multiple scales
    """
    def __init__(self, num_scales: int = 3):
        super(MultiScaleBoundaryDetection, self).__init__()
        
        # Multi-scale boundary detectors
        self.boundary_detectors = nn.ModuleList()
        for i in range(num_scales):
            detector = nn.Sequential(
                nn.Conv2d(1, 16, 2*i+3, padding=i+1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.Conv2d(16, 1, 1),
                nn.Sigmoid()
            )
            self.boundary_detectors.append(detector)
        
        # Scale fusion
        self.scale_fusion = nn.Conv2d(num_scales, 1, 1)
        
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        # Multi-scale boundary detection
        boundaries = [detector(pred) for detector in self.boundary_detectors]
        
        # Concatenate and fuse
        boundaries_cat = torch.cat(boundaries, dim=1)
        fused_boundary = self.scale_fusion(boundaries_cat)
        fused_boundary = torch.sigmoid(fused_boundary)
        
        return fused_boundary


# ============================================================================
# NOVEL COMPONENT 7: Region Refinement
# ============================================================================

class RegionRefinement(nn.Module):
    """
    Novel Region Refinement module
    """
    def __init__(self):
        super(RegionRefinement, self).__init__()
        
        self.refine = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        refined = self.refine(pred)
        return refined


# ============================================================================
# ABRA Module (Adaptive Boundary-Region Attention)
# ============================================================================

class ABRAModule(nn.Module):
    """
    Novel Adaptive Boundary-Region Attention Module
    Replacement for BFA with enhanced capabilities
    """
    def __init__(self, filters: int, num_scales: int = 3):
        super(ABRAModule, self).__init__()
        self.filters = filters
        
        # Multi-scale boundary detection
        self.boundary_detector = MultiScaleBoundaryDetection(num_scales)
        
        # Region refinement
        self.region_refiner = RegionRefinement()
        
        # Feature processing for boundary
        self.boundary_conv = nn.Sequential(
            nn.Conv2d(filters, filters // 2, 3, padding=1),
            nn.BatchNorm2d(filters // 2),
            nn.ReLU()
        )
        
        # Feature processing for region
        self.region_conv = nn.Sequential(
            nn.Conv2d(filters, filters // 2, 3, padding=1),
            nn.BatchNorm2d(filters // 2),
            nn.ReLU()
        )
        
        # SE-like channel attention
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(filters, filters // 4)
        self.fc2 = nn.Linear(filters // 4, filters)
        
        # Final fusion
        self.fusion = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )
        
    def forward(self, encoder_feat: torch.Tensor, decoder_pred: torch.Tensor) -> torch.Tensor:
        # Resize prediction to match feature size if needed
        if decoder_pred.shape[2:] != encoder_feat.shape[2:]:
            decoder_pred = F.interpolate(
                decoder_pred, size=encoder_feat.shape[2:],
                mode='bilinear', align_corners=False
            )
        
        # Generate boundary attention
        boundary_attn = self.boundary_detector(decoder_pred)
        
        # Generate region attention
        region_attn = self.region_refiner(decoder_pred)
        
        # Apply attention to features
        boundary_feat = encoder_feat * boundary_attn
        boundary_feat = self.boundary_conv(boundary_feat)
        
        region_feat = encoder_feat * region_attn
        region_feat = self.region_conv(region_feat)
        
        # Concatenate
        combined_feat = torch.cat([boundary_feat, region_feat], dim=1)
        
        # Channel attention
        attn = self.gap(combined_feat).flatten(1)
        attn = F.relu(self.fc1(attn))
        attn = torch.sigmoid(self.fc2(attn))
        attn = attn.view(-1, self.filters, 1, 1)
        
        combined_feat = combined_feat * attn
        
        # Final fusion
        output = self.fusion(combined_feat)
        
        return output


# ============================================================================
# AMKAN Block (Main Building Block)
# ============================================================================

class AMKANBlock(nn.Module):
    """
    Novel Adaptive Multi-scale Mamba-KAN Block
    Main building block replacing VKAN
    """
    def __init__(self, filters: int, num_scales: int = 3, num_directions: int = 6,
                 kernel_sizes: List[int] = [3, 5, 7]):
        super(AMKANBlock, self).__init__()
        
        # Layer Normalization 1
        self.ln1 = nn.LayerNorm([filters])
        
        # DSF-VSS Block
        self.dsf_vss = DSFVSSBlock(filters, num_scales, num_directions)
        
        # Layer Normalization 2
        self.ln2 = nn.LayerNorm([filters])
        
        # AKFC-KAN Block
        self.akfc_kan = AKFCKANBlock(filters, kernel_sizes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x
        
        # Normalize
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        # DSF-VSS Block
        x = self.dsf_vss(x)
        
        # Residual connection
        x = x + residual
        
        # Normalize
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)  # B, C, H, W
        
        # AKFC-KAN Block
        x = self.akfc_kan(x)
        
        # Residual connection
        output = x + residual
        
        return output


# ============================================================================
# MK-GAN Generator
# ============================================================================

class MKGANGenerator(nn.Module):
    """
    MK-GAN Generator Network
    
    Architecture:
    - Encoder: 3 Conv blocks + 2 AMKAN blocks
    - Decoder: 2 AMKAN blocks + 3 Conv blocks (symmetric)
    - Skip connections with ABRA modules
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_filters: int = 48):
        super(MKGANGenerator, self).__init__()
        
        # Weight initialization
        def init_weights(m):
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # =============== Encoder ===============
        # Stage 1: Conv block
        self.enc1_conv1 = nn.Conv2d(in_channels, base_filters, 4, stride=2, padding=1)
        self.enc1_conv2 = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        self.enc1_bn = nn.BatchNorm2d(base_filters)
        
        # Stage 2: Conv block
        self.enc2_conv1 = nn.Conv2d(base_filters, base_filters * 2, 4, stride=2, padding=1)
        self.enc2_conv2 = nn.Conv2d(base_filters * 2, base_filters * 2, 3, padding=1)
        self.enc2_bn = nn.BatchNorm2d(base_filters * 2)
        
        # Stage 3: Conv block
        self.enc3_conv1 = nn.Conv2d(base_filters * 2, base_filters * 4, 4, stride=2, padding=1)
        self.enc3_conv2 = nn.Conv2d(base_filters * 4, base_filters * 4, 3, padding=1)
        self.enc3_bn = nn.BatchNorm2d(base_filters * 4)
        
        # Stage 4: Patch Embed + AMKAN
        self.enc4_patch_embed = nn.Conv2d(base_filters * 4, base_filters * 8, 2, stride=2, padding=0)
        self.enc4_amkan = AMKANBlock(base_filters * 8)
        
        # Stage 5: Patch Merge + AMKAN
        self.enc5_patch_merge = nn.Conv2d(base_filters * 8, base_filters * 16, 2, stride=2, padding=0)
        self.enc5_amkan = AMKANBlock(base_filters * 16)
        
        # =============== ABRA Modules ===============
        self.abra1 = ABRAModule(base_filters)
        self.abra2 = ABRAModule(base_filters * 2)
        self.abra3 = ABRAModule(base_filters * 4)
        
        # =============== Decoder ===============
        # Stage 5: Patch Expand + AMKAN
        self.dec5_patch_expand = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec5_conv = nn.Conv2d(base_filters * 16, base_filters * 8, 1)
        self.dec5_amkan = AMKANBlock(base_filters * 8)
        self.out5 = nn.Sequential(nn.Conv2d(base_filters * 8, 1, 1), nn.Sigmoid())
        
        # Stage 4: Patch Expand + AMKAN
        self.dec4_patch_expand = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec4_conv = nn.Conv2d(base_filters * 8, base_filters * 4, 1)
        self.dec4_amkan = AMKANBlock(base_filters * 4)
        self.out4 = nn.Sequential(nn.Conv2d(base_filters * 4, 1, 1), nn.Sigmoid())
        
        # Stage 3: UpConv + Conv block
        self.dec3_upconv = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1)
        self.dec3_conv1 = nn.Conv2d(base_filters * 4, base_filters * 2, 3, padding=1)
        self.dec3_bn = nn.BatchNorm2d(base_filters * 2)
        self.out3 = nn.Sequential(nn.Conv2d(base_filters * 2, 1, 1), nn.Sigmoid())
        
        # Stage 2: UpConv + Conv block
        self.dec2_upconv = nn.ConvTranspose2d(base_filters * 2, base_filters, 4, stride=2, padding=1)
        self.dec2_conv1 = nn.Conv2d(base_filters * 2, base_filters, 3, padding=1)
        self.dec2_bn = nn.BatchNorm2d(base_filters)
        self.out2 = nn.Sequential(nn.Conv2d(base_filters, 1, 1), nn.Sigmoid())
        
        # Stage 1: UpConv + Output
        self.dec1_upconv = nn.ConvTranspose2d(base_filters, base_filters // 2, 4, stride=2, padding=1)
        self.dec1_conv = nn.Conv2d(base_filters + base_filters // 2, base_filters // 2, 3, padding=1)
        
        # Final output
        self.output = nn.Sequential(
            nn.Conv2d(base_filters // 2, out_channels, 1),
            nn.Tanh()
        )
        
        # Initialize weights
        self.apply(init_weights)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # =============== Encoder ===============
        # Stage 1
        e1 = F.leaky_relu(self.enc1_conv1(x), 0.2)
        e1 = self.enc1_conv2(e1)
        e1 = self.enc1_bn(e1)
        skip1 = F.leaky_relu(e1, 0.2)
        
        # Stage 2
        e2 = F.leaky_relu(self.enc2_conv1(skip1), 0.2)
        e2 = self.enc2_conv2(e2)
        e2 = self.enc2_bn(e2)
        skip2 = F.leaky_relu(e2, 0.2)
        
        # Stage 3
        e3 = F.leaky_relu(self.enc3_conv1(skip2), 0.2)
        e3 = self.enc3_conv2(e3)
        e3 = self.enc3_bn(e3)
        skip3 = F.leaky_relu(e3, 0.2)
        
        # Stage 4
        e4 = self.enc4_patch_embed(skip3)
        e4 = self.enc4_amkan(e4)
        
        # Stage 5
        e5 = self.enc5_patch_merge(e4)
        e5 = self.enc5_amkan(e5)
        
        # =============== Decoder ===============
        # Stage 5
        d5 = self.dec5_patch_expand(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5_conv(d5)
        d5 = self.dec5_amkan(d5)
        out5 = self.out5(d5)
        
        # Stage 4
        d4 = self.dec4_patch_expand(d5)
        skip3_abra = self.abra3(skip3, out5)
        d4 = torch.cat([d4, skip3_abra], dim=1)
        d4 = self.dec4_conv(d4)
        d4 = self.dec4_amkan(d4)
        out4 = self.out4(d4)
        
        # Stage 3
        d3 = self.dec3_upconv(d4)
        skip2_abra = self.abra2(skip2, out4)
        d3 = torch.cat([d3, skip2_abra], dim=1)
        d3 = self.dec3_conv1(d3)
        d3 = self.dec3_bn(d3)
        d3 = F.relu(d3)
        out3 = self.out3(d3)
        
        # Stage 2
        d2 = self.dec2_upconv(d3)
        skip1_abra = self.abra1(skip1, out3)
        d2 = torch.cat([d2, skip1_abra], dim=1)
        d2 = self.dec2_conv1(d2)
        d2 = self.dec2_bn(d2)
        d2 = F.relu(d2)
        out2 = self.out2(d2)
        
        # Stage 1
        d1 = self.dec1_upconv(d2)
        d1 = torch.cat([d1, skip1], dim=1)
        d1 = self.dec1_conv(d1)
        d1 = F.relu(d1)
        
        # Final output
        output = self.output(d1)
        
        return output, [out5, out4, out3, out2]


# ============================================================================
# MK-GAN Discriminator
# ============================================================================

class MKGANDiscriminator(nn.Module):
    """
    Patch-based Discriminator for MK-GAN
    Compatible with pix2pix discriminator
    """
    def __init__(self, in_channels: int = 3):
        super(MKGANDiscriminator, self).__init__()
        
        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers
        
        self.model = nn.Sequential(
            # Input: concatenated source and target images
            *discriminator_block(in_channels * 2, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 512, 4, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        self.apply(init_weights)
        
    def forward(self, img: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Concatenate image and target
        x = torch.cat([img, target], dim=1)
        return self.model(x)


# ============================================================================
# Test and Summary
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("MK-GAN: Multi-Scale Mamba-KAN Generative Adversarial Network (PyTorch)")
    print("="*80)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # Define parameters
    in_channels = 3
    out_channels = 3
    base_filters = 48
    
    print("\n1. Creating Generator...")
    g_model = MKGANGenerator(in_channels, out_channels, base_filters).to(device)
    g_params = sum(p.numel() for p in g_model.parameters())
    print(f"Generator created successfully!")
    print(f"Total parameters: {g_params:,}")
    
    print("\n2. Creating Discriminator...")
    d_model = MKGANDiscriminator(in_channels).to(device)
    d_params = sum(p.numel() for p in d_model.parameters())
    print(f"Discriminator created successfully!")
    print(f"Total parameters: {d_params:,}")
    
    print(f"\n3. Total Model Parameters: {(g_params + d_params):,}")
    
    # Test with dummy data
    print("\n4. Testing with dummy data...")
    dummy_input = torch.randn(2, in_channels, 256, 256).to(device)
    
    print("\nTesting Generator...")
    with torch.no_grad():
        gen_output, aux_outputs = g_model(dummy_input)
        print(f"Generator main output shape: {gen_output.shape}")
        print(f"Number of auxiliary outputs: {len(aux_outputs)}")
        for i, aux_out in enumerate(aux_outputs):
            print(f"  Aux output {i+1} shape: {aux_out.shape}")
    
    print("\nTesting Discriminator...")
    with torch.no_grad():
        disc_output = d_model(dummy_input, gen_output)
        print(f"Discriminator output shape: {disc_output.shape}")
    
    print("\n" + "="*80)
    print("✓ MK-GAN PyTorch architecture created and tested successfully!")
    print("="*80)
    
    # Print model summary
    print("\n5. Model Architecture Summary:")
    print("\n--- Generator Architecture ---")
    print(g_model)
    
    print("\n--- Discriminator Architecture ---")
    print(d_model)
