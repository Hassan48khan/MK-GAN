"""
MK-GAN: Multi-Scale Mamba-KAN Generative Adversarial Network
Complete Architecture Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List


# ============================================================================
# Dynamic Scale Fusion (DSF)
# ============================================================================

class DynamicScaleFusion(nn.Module):
    """Dynamic Scale Fusion with learnable weights"""
    def __init__(self, filters: int, num_scales: int = 3):
        super(DynamicScaleFusion, self).__init__()
        self.filters = filters
        self.num_scales = num_scales

        self.scale_convs = nn.ModuleList([
            nn.Conv2d(filters, filters, kernel_size=2*i+3, padding=i+1)
            for i in range(num_scales)
        ])

        self.scale_weights = nn.Parameter(torch.ones(num_scales, 1, 1, 1))
        self.fusion_conv = nn.Conv2d(filters * num_scales, filters, 1)
        self.ln = nn.LayerNorm([filters])
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale_features = []
        for i, conv in enumerate(self.scale_convs):
            feat = conv(x)
            weighted_feat = feat * self.scale_weights[i]
            scale_features.append(weighted_feat)

        fused = torch.cat(scale_features, dim=1)
        output = self.fusion_conv(fused)

        output = output.permute(0, 2, 3, 1)
        output = self.ln(output)
        output = output.permute(0, 3, 1, 2)
        output = self.act(output)

        return output


# ============================================================================
# Learnable Directional Scanning (LDS)
# ============================================================================

class LearnableDirectionalScanning(nn.Module):
    """Six-directional learnable scanning"""
    def __init__(self, filters: int, num_directions: int = 6):
        super(LearnableDirectionalScanning, self).__init__()
        self.filters = filters
        self.num_directions = num_directions

        self.dir_convs = nn.ModuleList([
            nn.Conv2d(filters, filters, 3, padding=1)
            for _ in range(num_directions)
        ])

        self.dir_embed = nn.Parameter(torch.randn(num_directions, filters))
        self.attention_conv = nn.Conv2d(num_directions, num_directions, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape

        dir_features = []
        for i, conv in enumerate(self.dir_convs):
            dir_feat = conv(x)
            dir_emb = self.dir_embed[i].view(1, C, 1, 1)
            dir_feat = dir_feat + dir_emb
            dir_features.append(dir_feat)

        dir_stack = torch.stack(dir_features, dim=1)
        attn_input = dir_stack.mean(dim=2)
        attn_weights = self.attention_conv(attn_input)
        attn_weights = F.softmax(attn_weights, dim=1)

        attn_weights = attn_weights.unsqueeze(2)
        output = (dir_stack * attn_weights).sum(dim=1)

        return output


# ============================================================================
# S6 State-Space Block
# ============================================================================

class S6Block(nn.Module):
    """State Space Block for 2D feature modeling"""
    def __init__(self, filters: int, state_dim: int = 16):
        super(S6Block, self).__init__()
        self.conv_in = nn.Conv2d(filters, state_dim, 1)
        self.conv_state = nn.Conv2d(state_dim, state_dim, 3, padding=1)
        self.conv_out = nn.Conv2d(state_dim, filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.conv_in(x)
        state = self.conv_state(state)
        state = F.relu(state)
        output = self.conv_out(state)
        return output


# ============================================================================
# DSF-VSS Block
# ============================================================================

class DSFVSSBlock(nn.Module):
    """Dynamic Scale Fusion VSS Block"""
    def __init__(self, filters: int, num_scales: int = 3, num_directions: int = 6):
        super(DSFVSSBlock, self).__init__()
        self.filters = filters

        self.ln1 = nn.LayerNorm([filters])
        self.expand = nn.Conv2d(filters, filters * 4, 1)
        self.dsf = DynamicScaleFusion(filters * 2, num_scales)
        self.dwconv = nn.Conv2d(filters * 2, filters * 2, 3, padding=1, groups=filters * 2)
        self.lds = LearnableDirectionalScanning(filters * 2, num_directions)
        self.s6 = S6Block(filters * 2)
        self.proj = nn.Conv2d(filters * 2, filters, 1)
        self.ln2 = nn.LayerNorm([filters * 2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = x.permute(0, 2, 3, 1)
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.expand(x)
        x1, x2 = torch.chunk(x, 2, dim=1)

        x1 = self.dsf(x1)
        x1 = self.dwconv(x1)
        x1 = F.silu(x1)
        x1 = self.lds(x1)
        x1 = self.s6(x1)

        x1 = x1.permute(0, 2, 3, 1)
        x1 = self.ln2(x1)
        x1 = x1.permute(0, 3, 1, 2)

        x2 = F.silu(x2)
        x_out = x1 * x2
        x_out = self.proj(x_out)

        output = x_out + residual
        return output


# ============================================================================
# Adaptive Kernel Fusion (AKF)
# ============================================================================

class AdaptiveKernelFusion(nn.Module):
    """Adaptive Kernel Fusion with multiple kernel sizes"""
    def __init__(self, filters: int, kernel_sizes: List[int] = [3, 5, 7]):
        super(AdaptiveKernelFusion, self).__init__()
        self.filters = filters
        self.kernel_sizes = kernel_sizes

        self.kernel_branches = nn.ModuleList()
        for k in kernel_sizes:
            branch = nn.Sequential(
                nn.Conv2d(filters, filters, k, padding=k//2, groups=filters),
                nn.BatchNorm2d(filters),
                nn.SiLU()
            )
            self.kernel_branches.append(branch)

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(filters * len(kernel_sizes), filters // 4)
        self.fc2 = nn.Linear(filters // 4, len(kernel_sizes))
        self.fusion = nn.Conv2d(filters, filters, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.kernel_branches]
        concat = torch.cat(branch_outputs, dim=1)

        attn = self.gap(concat).flatten(1)
        attn = F.silu(self.fc1(attn))
        attn = F.softmax(self.fc2(attn), dim=1)

        attn = attn.view(-1, len(self.kernel_sizes), 1, 1, 1)
        branch_stack = torch.stack(branch_outputs, dim=1)
        fused = (branch_stack * attn).sum(dim=1)

        output = self.fusion(fused)
        return output


# ============================================================================
# Tok-KAN Block
# ============================================================================

class TokKAN(nn.Module):
    """Token-based KAN block"""
    def __init__(self, filters: int, expansion: int = 2):
        super(TokKAN, self).__init__()
        hidden = filters * expansion

        self.conv1 = nn.Conv2d(filters, hidden, 1)
        self.dwconv1 = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.bn1 = nn.BatchNorm2d(hidden)
        self.conv2 = nn.Conv2d(hidden, filters, 1)
        self.dwconv2 = nn.Conv2d(filters, filters, 3, padding=1, groups=filters)
        self.bn2 = nn.BatchNorm2d(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = F.silu(out)
        out = self.dwconv1(out)
        out = self.bn1(out)

        out = self.conv2(out)
        out = F.silu(out)
        out = self.dwconv2(out)
        out = self.bn2(out)

        return out


# ============================================================================
# AKFC-KAN Block
# ============================================================================

class AKFCKANBlock(nn.Module):
    """Adaptive Kernel Fusion Convolutional KAN Block"""
    def __init__(self, filters: int, kernel_sizes: List[int] = [3, 5, 7]):
        super(AKFCKANBlock, self).__init__()

        self.akf = AdaptiveKernelFusion(filters, kernel_sizes)
        self.ln = nn.LayerNorm([filters])
        self.tok_kan = TokKAN(filters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = self.akf(x)

        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)

        x = self.tok_kan(x)
        output = x + residual
        return output


# ============================================================================
# Multi-Scale Boundary Detection
# ============================================================================

class MultiScaleBoundaryDetection(nn.Module):
    """Multi-scale boundary detection module"""
    def __init__(self, num_scales: int = 3):
        super(MultiScaleBoundaryDetection, self).__init__()

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

        self.scale_fusion = nn.Conv2d(num_scales, 1, 1)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        boundaries = [detector(pred) for detector in self.boundary_detectors]
        boundaries_cat = torch.cat(boundaries, dim=1)
        fused_boundary = self.scale_fusion(boundaries_cat)
        fused_boundary = torch.sigmoid(fused_boundary)
        return fused_boundary


# ============================================================================
# Region Refinement
# ============================================================================

class RegionRefinement(nn.Module):
    """Region refinement network"""
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
        return self.refine(pred)


# ============================================================================
# ABRA Module
# ============================================================================

class ABRAModule(nn.Module):
    """Adaptive Boundary-Region Attention Module"""
    def __init__(self, filters: int, num_scales: int = 3):
        super(ABRAModule, self).__init__()
        self.filters = filters

        self.boundary_detector = MultiScaleBoundaryDetection(num_scales)
        self.region_refiner = RegionRefinement()

        self.boundary_conv = nn.Sequential(
            nn.Conv2d(filters, filters // 2, 3, padding=1),
            nn.BatchNorm2d(filters // 2),
            nn.ReLU()
        )

        self.region_conv = nn.Sequential(
            nn.Conv2d(filters, filters // 2, 3, padding=1),
            nn.BatchNorm2d(filters // 2),
            nn.ReLU()
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(filters, filters // 4)
        self.fc2 = nn.Linear(filters // 4, filters)

        self.fusion = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1),
            nn.BatchNorm2d(filters),
            nn.ReLU()
        )

    def forward(self, encoder_feat: torch.Tensor, decoder_pred: torch.Tensor) -> torch.Tensor:
        if decoder_pred.shape[2:] != encoder_feat.shape[2:]:
            decoder_pred = F.interpolate(
                decoder_pred, size=encoder_feat.shape[2:],
                mode='bilinear', align_corners=False
            )

        boundary_attn = self.boundary_detector(decoder_pred)
        region_attn = self.region_refiner(decoder_pred)

        boundary_feat = encoder_feat * boundary_attn
        boundary_feat = self.boundary_conv(boundary_feat)

        region_feat = encoder_feat * region_attn
        region_feat = self.region_conv(region_feat)

        combined_feat = torch.cat([boundary_feat, region_feat], dim=1)

        attn = self.gap(combined_feat).flatten(1)
        attn = F.relu(self.fc1(attn))
        attn = torch.sigmoid(self.fc2(attn))
        attn = attn.view(-1, self.filters, 1, 1)

        combined_feat = combined_feat * attn
        output = self.fusion(combined_feat)

        return output


# ============================================================================
# AMKAN Block
# ============================================================================

class AMKANBlock(nn.Module):
    """Adaptive Multi-scale Mamba-KAN Block"""
    def __init__(self, filters: int, num_scales: int = 3, num_directions: int = 6,
                 kernel_sizes: List[int] = [3, 5, 7]):
        super(AMKANBlock, self).__init__()

        self.ln1 = nn.LayerNorm([filters])
        self.dsf_vss = DSFVSSBlock(filters, num_scales, num_directions)
        self.ln2 = nn.LayerNorm([filters])
        self.akfc_kan = AKFCKANBlock(filters, kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        x = x.permute(0, 2, 3, 1)
        x = self.ln1(x)
        x = x.permute(0, 3, 1, 2)

        x = self.dsf_vss(x)
        x = x + residual

        x = x.permute(0, 2, 3, 1)
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)

        x = self.akfc_kan(x)
        output = x + residual

        return output


# ============================================================================
# MK-GAN Generator
# ============================================================================

class MKGANGenerator(nn.Module):
    """MK-GAN Generator with AMKAN blocks and ABRA modules"""
    def __init__(self, in_channels: int = 3, out_channels: int = 3, base_filters: int = 48):
        super(MKGANGenerator, self).__init__()

        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Encoder
        self.enc1_conv1 = nn.Conv2d(in_channels, base_filters, 4, stride=2, padding=1)
        self.enc1_conv2 = nn.Conv2d(base_filters, base_filters, 3, padding=1)
        self.enc1_bn = nn.BatchNorm2d(base_filters)

        self.enc2_conv1 = nn.Conv2d(base_filters, base_filters * 2, 4, stride=2, padding=1)
        self.enc2_conv2 = nn.Conv2d(base_filters * 2, base_filters * 2, 3, padding=1)
        self.enc2_bn = nn.BatchNorm2d(base_filters * 2)

        self.enc3_conv1 = nn.Conv2d(base_filters * 2, base_filters * 4, 4, stride=2, padding=1)
        self.enc3_conv2 = nn.Conv2d(base_filters * 4, base_filters * 4, 3, padding=1)
        self.enc3_bn = nn.BatchNorm2d(base_filters * 4)

        self.enc4_patch_embed = nn.Conv2d(base_filters * 4, base_filters * 8, 2, stride=2, padding=0)
        self.enc4_amkan = AMKANBlock(base_filters * 8)

        self.enc5_patch_merge = nn.Conv2d(base_filters * 8, base_filters * 16, 2, stride=2, padding=0)
        self.enc5_amkan = AMKANBlock(base_filters * 16)

        # ABRA Modules
        self.abra1 = ABRAModule(base_filters)
        self.abra2 = ABRAModule(base_filters * 2)
        self.abra3 = ABRAModule(base_filters * 4)

        # Decoder
        self.dec5_patch_expand = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, 2, stride=2)
        self.dec5_conv = nn.Conv2d(base_filters * 16, base_filters * 8, 1)
        self.dec5_amkan = AMKANBlock(base_filters * 8)
        self.out5 = nn.Sequential(nn.Conv2d(base_filters * 8, 1, 1), nn.Sigmoid())

        self.dec4_patch_expand = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, 2, stride=2)
        self.dec4_conv = nn.Conv2d(base_filters * 8, base_filters * 4, 1)
        self.dec4_amkan = AMKANBlock(base_filters * 4)
        self.out4 = nn.Sequential(nn.Conv2d(base_filters * 4, 1, 1), nn.Sigmoid())

        self.dec3_upconv = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, 4, stride=2, padding=1)
        self.dec3_conv1 = nn.Conv2d(base_filters * 4, base_filters * 2, 3, padding=1)
        self.dec3_bn = nn.BatchNorm2d(base_filters * 2)
        self.out3 = nn.Sequential(nn.Conv2d(base_filters * 2, 1, 1), nn.Sigmoid())

        self.dec2_upconv = nn.ConvTranspose2d(base_filters * 2, base_filters, 4, stride=2, padding=1)
        self.dec2_conv1 = nn.Conv2d(base_filters * 2, base_filters, 3, padding=1)
        self.dec2_bn = nn.BatchNorm2d(base_filters)
        self.out2 = nn.Sequential(nn.Conv2d(base_filters, 1, 1), nn.Sigmoid())

        self.dec1_upconv = nn.ConvTranspose2d(base_filters, base_filters // 2, 4, stride=2, padding=1)
        self.dec1_conv_final = nn.Conv2d(base_filters // 2, base_filters // 2, 3, padding=1)

        self.output = nn.Sequential(
            nn.Conv2d(base_filters // 2, out_channels, 1),
            nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # Encoder
        e1 = F.leaky_relu(self.enc1_conv1(x), 0.2)
        e1 = self.enc1_conv2(e1)
        e1 = self.enc1_bn(e1)
        skip1 = F.leaky_relu(e1, 0.2)

        e2 = F.leaky_relu(self.enc2_conv1(skip1), 0.2)
        e2 = self.enc2_conv2(e2)
        e2 = self.enc2_bn(e2)
        skip2 = F.leaky_relu(e2, 0.2)

        e3 = F.leaky_relu(self.enc3_conv1(skip2), 0.2)
        e3 = self.enc3_conv2(e3)
        e3 = self.enc3_bn(e3)
        skip3 = F.leaky_relu(e3, 0.2)

        e4 = self.enc4_patch_embed(skip3)
        e4 = self.enc4_amkan(e4)

        e5 = self.enc5_patch_merge(e4)
        e5 = self.enc5_amkan(e5)

        # Decoder
        d5 = self.dec5_patch_expand(e5)
        d5 = torch.cat([d5, e4], dim=1)
        d5 = self.dec5_conv(d5)
        d5 = self.dec5_amkan(d5)
        out5 = self.out5(d5)

        d4 = self.dec4_patch_expand(d5)
        skip3_abra = self.abra3(skip3, out5)
        d4 = torch.cat([d4, skip3_abra], dim=1)
        d4 = self.dec4_conv(d4)
        d4 = self.dec4_amkan(d4)
        out4 = self.out4(d4)

        d3 = self.dec3_upconv(d4)
        skip2_abra = self.abra2(skip2, out4)
        d3 = torch.cat([d3, skip2_abra], dim=1)
        d3 = self.dec3_conv1(d3)
        d3 = self.dec3_bn(d3)
        d3 = F.relu(d3)
        out3 = self.out3(d3)

        d2 = self.dec2_upconv(d3)
        skip1_abra = self.abra1(skip1, out3)
        d2 = torch.cat([d2, skip1_abra], dim=1)
        d2 = self.dec2_conv1(d2)
        d2 = self.dec2_bn(d2)
        d2 = F.relu(d2)
        out2 = self.out2(d2)

        d1 = self.dec1_upconv(d2)
        d1 = self.dec1_conv_final(d1)
        d1 = F.relu(d1)

        output = self.output(d1)

        return output, [out5, out4, out3, out2]


# ============================================================================
# MK-GAN Discriminator
# ============================================================================

class MKGANDiscriminator(nn.Module):
    """PatchGAN Discriminator"""
    def __init__(self, in_channels: int = 3):
        super(MKGANDiscriminator, self).__init__()

        def discriminator_block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
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

        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.apply(init_weights)

    def forward(self, img: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        x = torch.cat([img, target], dim=1)
        return self.model(x)


# ============================================================================
# Main Test
# ============================================================================

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    generator = MKGANGenerator(3, 3, 48).to(device)
    discriminator = MKGANDiscriminator(3).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())

    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    print(f"Total parameters: {(g_params + d_params):,}")

    # Test forward pass
    dummy_input = torch.randn(2, 3, 256, 256).to(device)
    with torch.no_grad():
        gen_output, aux_outputs = generator(dummy_input)
        disc_output = discriminator(dummy_input, gen_output)

    print(f"\nGenerator output shape: {gen_output.shape}")
    print(f"Auxiliary outputs: {len(aux_outputs)}")
    print(f"Discriminator output shape: {disc_output.shape}")
    print("\n✓ MK-GAN architecture ready!")
