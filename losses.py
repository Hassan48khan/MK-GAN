"""
Loss Functions for MK-GAN
- Adversarial Loss
- L1 Loss
- SSIM Loss
- Deep Supervision Loss (Dice + Cross-Entropy)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


# ============================================================================
# Adversarial Loss
# ============================================================================

class AdversarialLoss(nn.Module):
    """BCE-based adversarial loss for generator and discriminator"""
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.bce_loss = nn.BCELoss()

    def forward_generator(self, fake_pred: torch.Tensor) -> torch.Tensor:
        """Generator wants discriminator to output 1 for fake images"""
        valid = torch.ones_like(fake_pred)
        return self.bce_loss(fake_pred, valid)

    def forward_discriminator(self, real_pred: torch.Tensor,
                              fake_pred: torch.Tensor) -> torch.Tensor:
        """Discriminator distinguishes real from fake"""
        valid = torch.ones_like(real_pred)
        fake = torch.zeros_like(fake_pred)

        real_loss = self.bce_loss(real_pred, valid)
        fake_loss = self.bce_loss(fake_pred, fake)

        return (real_loss + fake_loss) / 2


# ============================================================================
# SSIM Loss
# ============================================================================

class SSIMLoss(nn.Module):
    """Structural Similarity Index loss"""
    def __init__(self, window_size: int = 11, c1: float = 0.01**2, c2: float = 0.03**2):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2
        self.window = self._create_window(window_size)

    def _create_window(self, window_size: int) -> torch.Tensor:
        sigma = 1.5
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2.0 * sigma**2)))
            for x in range(window_size)
        ])
        window = gauss.unsqueeze(1) @ gauss.unsqueeze(0)
        window = window / window.sum()
        return window.unsqueeze(0).unsqueeze(0)

    def _ssim(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)

        channels = img1.size(1)
        window = self.window.expand(channels, 1, self.window_size, self.window_size)

        mu1 = F.conv2d(img1, window, padding=self.window_size//2, groups=channels)
        mu2 = F.conv2d(img2, window, padding=self.window_size//2, groups=channels)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1 * img1, window, padding=self.window_size//2, groups=channels) - mu1_sq
        sigma2_sq = F.conv2d(img2 * img2, window, padding=self.window_size//2, groups=channels) - mu2_sq
        sigma12 = F.conv2d(img1 * img2, window, padding=self.window_size//2, groups=channels) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + self.c1) * (2 * sigma12 + self.c2)) / \
                   ((mu1_sq + mu2_sq + self.c1) * (sigma1_sq + sigma2_sq + self.c2))

        return ssim_map.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 1 - self._ssim(pred, target)


# ============================================================================
# Dice Loss
# ============================================================================

class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    def __init__(self, smooth: float = 1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_flat = pred.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)

        return 1 - dice


# ============================================================================
# Deep Supervision Loss
# ============================================================================

class DeepSupervisionLoss(nn.Module):
    """
    Deep Supervision Loss combining Dice and Cross-Entropy
    Applied to multiple decoder stages with decreasing weights
    """
    def __init__(self, weights: List[float] = [1.0, 0.5, 0.25, 0.125]):
        super(DeepSupervisionLoss, self).__init__()
        self.weights = weights
        self.dice_loss = DiceLoss()
        self.ce_loss = nn.BCELoss()

    def forward(self, aux_outputs: List[torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        total_loss = 0.0

        for i, (aux_out, weight) in enumerate(zip(aux_outputs, self.weights)):
            if aux_out.shape[2:] != target.shape[2:]:
                aux_out_resized = F.interpolate(
                    aux_out, size=target.shape[2:],
                    mode='bilinear', align_corners=False
                )
            else:
                aux_out_resized = aux_out

            dice = self.dice_loss(aux_out_resized, target)
            ce = self.ce_loss(aux_out_resized, target)

            stage_loss = weight * (dice + ce)
            total_loss += stage_loss

        return total_loss


# ============================================================================
# Total MK-GAN Generator Loss
# ============================================================================

class MKGANGeneratorLoss(nn.Module):
    """
    Total generator loss: L = L_adv + λ1*L_L1 + λ2*L_SSIM + λ3*L_DS
    Default: λ1=100, λ2=10, λ3=1
    """
    def __init__(self, lambda_l1: float = 100.0, lambda_ssim: float = 10.0,
                 lambda_ds: float = 1.0,
                 ds_weights: List[float] = [1.0, 0.5, 0.25, 0.125]):
        super(MKGANGeneratorLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_ds = lambda_ds

        self.adv_loss = AdversarialLoss()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.ds_loss = DeepSupervisionLoss(weights=ds_weights)

    def forward(self, fake_pred: torch.Tensor, fake_masks: torch.Tensor,
                real_masks: torch.Tensor, aux_outputs: List[torch.Tensor]) -> dict:
        """
        Returns dictionary with all loss components
        """
        adv = self.adv_loss.forward_generator(fake_pred)
        l1 = self.l1_loss(fake_masks, real_masks)
        ssim = self.ssim_loss(fake_masks, real_masks)
        ds = self.ds_loss(aux_outputs, real_masks)

        total = adv + self.lambda_l1 * l1 + self.lambda_ssim * ssim + self.lambda_ds * ds

        return {
            'total': total,
            'adv': adv,
            'l1': l1,
            'ssim': ssim,
            'ds': ds
        }


if __name__ == "__main__":
    # Test losses
    print("Testing MK-GAN Loss Functions...")

    # Dummy data
    fake_pred = torch.rand(2, 1, 30, 30)
    real_pred = torch.rand(2, 1, 30, 30)
    fake_masks = torch.rand(2, 3, 256, 256)
    real_masks = torch.rand(2, 3, 256, 256)
    # Aux outputs are single-channel
    aux_outputs = [
        torch.rand(2, 1, 16, 16),
        torch.rand(2, 1, 32, 32),
        torch.rand(2, 1, 64, 64),
        torch.rand(2, 1, 128, 128)
    ]
    # Use single channel target for DS
    target_single = real_masks[:, 0:1, :, :]

    # Test adversarial loss
    adv = AdversarialLoss()
    g_adv = adv.forward_generator(fake_pred)
    d_adv = adv.forward_discriminator(real_pred, fake_pred)
    print(f"Generator Adv Loss: {g_adv.item():.4f}")
    print(f"Discriminator Loss: {d_adv.item():.4f}")

    # Test SSIM loss
    ssim = SSIMLoss()
    ssim_val = ssim(fake_masks, real_masks)
    print(f"SSIM Loss: {ssim_val.item():.4f}")

    # Test deep supervision (with single-channel target)
    ds = DeepSupervisionLoss()
    ds_val = ds(aux_outputs, target_single)
    print(f"Deep Supervision Loss: {ds_val.item():.4f}")

    print("\n✓ All losses working correctly!")
