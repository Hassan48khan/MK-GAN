"""
Training Script for MK-GAN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from MKGAN import MKGANGenerator, MKGANDiscriminator
from losses import AdversarialLoss, SSIMLoss, DeepSupervisionLoss
from dataloader import get_dataloader


# ============================================================================
# Metric Calculation
# ============================================================================

def calculate_dice(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate Dice coefficient"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum()

    if union == 0:
        return 1.0 if target_bin.sum() == 0 else 0.0

    dice = (2.0 * intersection) / (union + 1e-8)
    return dice.item()


def calculate_iou(pred: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    """Calculate IoU"""
    pred_bin = (pred > threshold).float()
    target_bin = (target > threshold).float()

    intersection = (pred_bin * target_bin).sum()
    union = pred_bin.sum() + target_bin.sum() - intersection

    if union == 0:
        return 1.0 if target_bin.sum() == 0 else 0.0

    iou = intersection / (union + 1e-8)
    return iou.item()


# ============================================================================
# Training Functions
# ============================================================================

def train_one_epoch(generator, discriminator, dataloader,
                    g_optimizer, d_optimizer,
                    adv_loss, ssim_loss, ds_loss, l1_loss,
                    lambda_l1, lambda_ssim, lambda_ds,
                    device, epoch, total_epochs):
    """Train for one epoch"""
    generator.train()
    discriminator.train()

    epoch_g_loss = 0
    epoch_d_loss = 0
    epoch_dice = 0
    epoch_iou = 0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs} [Train]')

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device)
        masks = masks.to(device)

        # ============== Train Discriminator ==============
        d_optimizer.zero_grad()

        # Generate fake masks
        fake_masks, aux_outputs = generator(images)

        # Discriminator predictions
        real_pred = discriminator(images, masks)
        fake_pred = discriminator(images, fake_masks.detach())

        # Discriminator loss
        d_loss = adv_loss.forward_discriminator(real_pred, fake_pred)

        d_loss.backward()
        d_optimizer.step()

        # ============== Train Generator ==============
        g_optimizer.zero_grad()

        # Adversarial loss
        fake_pred = discriminator(images, fake_masks)
        g_adv = adv_loss.forward_generator(fake_pred)

        # L1 loss
        g_l1 = l1_loss(fake_masks, masks)

        # SSIM loss
        g_ssim = ssim_loss(fake_masks, masks)

        # Deep supervision loss
        # Use first channel of mask for binary DS supervision
        target_for_ds = masks[:, 0:1, :, :]
        g_ds = ds_loss(aux_outputs, target_for_ds)

        # Total generator loss
        g_loss = g_adv + lambda_l1 * g_l1 + lambda_ssim * g_ssim + lambda_ds * g_ds

        g_loss.backward()
        g_optimizer.step()

        # Calculate metrics
        with torch.no_grad():
            dice = calculate_dice(fake_masks, masks)
            iou = calculate_iou(fake_masks, masks)

        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()
        epoch_dice += dice
        epoch_iou += iou

        pbar.set_postfix({
            'G': f'{g_loss.item():.3f}',
            'D': f'{d_loss.item():.3f}',
            'Dice': f'{dice:.3f}',
            'IoU': f'{iou:.3f}'
        })

    n = len(dataloader)
    return {
        'g_loss': epoch_g_loss / n,
        'd_loss': epoch_d_loss / n,
        'dice': epoch_dice / n,
        'iou': epoch_iou / n
    }


def validate(generator, dataloader, device):
    """Validate the model"""
    generator.eval()

    total_dice = 0
    total_iou = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            outputs, _ = generator(images)

            dice = calculate_dice(outputs, masks)
            iou = calculate_iou(outputs, masks)

            total_dice += dice
            total_iou += iou

            pbar.set_postfix({'Dice': f'{dice:.3f}', 'IoU': f'{iou:.3f}'})

    n = len(dataloader)
    return {'dice': total_dice / n, 'iou': total_iou / n}


def save_checkpoint(state: dict, path: str):
    """Save model checkpoint"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def adjust_learning_rate(optimizer, epoch, initial_lr, decay_epoch=100, decay_rate=0.5):
    """Decay learning rate by decay_rate every decay_epoch epochs"""
    lr = initial_lr * (decay_rate ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train MK-GAN')

    # Data arguments
    parser.add_argument('--train_image_dir', type=str, required=True,
                        help='Training images directory')
    parser.add_argument('--train_mask_dir', type=str, required=True,
                        help='Training masks directory')
    parser.add_argument('--val_image_dir', type=str, required=True,
                        help='Validation images directory')
    parser.add_argument('--val_mask_dir', type=str, required=True,
                        help='Validation masks directory')

    # Model arguments
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--base_filters', type=int, default=48)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)

    # Loss weights
    parser.add_argument('--lambda_l1', type=float, default=100.0)
    parser.add_argument('--lambda_ssim', type=float, default=10.0)
    parser.add_argument('--lambda_ds', type=float, default=1.0)

    # Checkpointing
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--save_freq', type=int, default=10)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--resume', type=str, default=None)

    # Early stopping
    parser.add_argument('--patience', type=int, default=20)

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Training MK-GAN for {args.epochs} epochs")

    # Create models
    print("\nCreating models...")
    generator = MKGANGenerator(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_filters=args.base_filters
    ).to(device)

    discriminator = MKGANDiscriminator(in_channels=args.in_channels).to(device)

    g_params = sum(p.numel() for p in generator.parameters())
    d_params = sum(p.numel() for p in discriminator.parameters())
    print(f"Generator parameters: {g_params:,}")
    print(f"Discriminator parameters: {d_params:,}")
    print(f"Total parameters: {(g_params + d_params):,}")

    # Loss functions
    adv_loss = AdversarialLoss()
    ssim_loss = SSIMLoss()
    ds_loss = DeepSupervisionLoss()
    l1_loss = nn.L1Loss()

    # Optimizers
    g_optimizer = optim.Adam(
        generator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )

    d_optimizer = optim.Adam(
        discriminator.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2)
    )

    # Data loaders
    print("\nLoading data...")
    train_loader = get_dataloader(
        image_dir=args.train_image_dir,
        mask_dir=args.train_mask_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        mode='train',
        shuffle=True
    )

    val_loader = get_dataloader(
        image_dir=args.val_image_dir,
        mask_dir=args.val_mask_dir,
        batch_size=1,
        img_size=args.img_size,
        num_workers=args.num_workers,
        mode='val',
        shuffle=False
    )

    # Resume from checkpoint
    start_epoch = 0
    best_dice = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nResuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer'])
        d_optimizer.load_state_dict(checkpoint['d_optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_dice = checkpoint.get('best_dice', 0)
        print(f"Resumed from epoch {start_epoch} (best Dice: {best_dice:.4f})")

    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)

    patience_counter = 0
    history = {'train_loss': [], 'val_dice': [], 'val_iou': []}

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()

        # Adjust learning rate
        current_lr = adjust_learning_rate(g_optimizer, epoch, args.lr)
        adjust_learning_rate(d_optimizer, epoch, args.lr)

        # Train
        train_metrics = train_one_epoch(
            generator, discriminator, train_loader,
            g_optimizer, d_optimizer,
            adv_loss, ssim_loss, ds_loss, l1_loss,
            args.lambda_l1, args.lambda_ssim, args.lambda_ds,
            device, epoch, args.epochs
        )

        # Validate
        if (epoch + 1) % args.val_freq == 0:
            val_metrics = validate(generator, val_loader, device)

            print(f"\n[Epoch {epoch+1}] LR: {current_lr:.6f}")
            print(f"Train - G_loss: {train_metrics['g_loss']:.4f}, "
                  f"D_loss: {train_metrics['d_loss']:.4f}, "
                  f"Dice: {train_metrics['dice']:.4f}")
            print(f"Val - Dice: {val_metrics['dice']:.4f}, "
                  f"IoU: {val_metrics['iou']:.4f}")

            # Save best model
            if val_metrics['dice'] > best_dice:
                best_dice = val_metrics['dice']
                patience_counter = 0

                save_checkpoint({
                    'epoch': epoch,
                    'generator': generator.state_dict(),
                    'discriminator': discriminator.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'best_dice': best_dice,
                    'args': vars(args)
                }, os.path.join(args.checkpoint_dir, 'mkgan_best.pth'))

                print(f"✓ Saved best model (Dice: {best_dice:.4f})")
            else:
                patience_counter += 1

            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {args.patience} epochs without improvement")
                break

            history['val_dice'].append(val_metrics['dice'])
            history['val_iou'].append(val_metrics['iou'])

        history['train_loss'].append(train_metrics['g_loss'])

        # Periodic checkpoint
        if (epoch + 1) % args.save_freq == 0:
            save_checkpoint({
                'epoch': epoch,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'g_optimizer': g_optimizer.state_dict(),
                'd_optimizer': d_optimizer.state_dict(),
                'best_dice': best_dice,
                'args': vars(args)
            }, os.path.join(args.checkpoint_dir, f'mkgan_epoch_{epoch+1}.pth'))

        epoch_time = time.time() - start_time
        print(f"Epoch time: {epoch_time:.1f}s")

    print("\n" + "="*60)
    print(f"Training completed! Best validation Dice: {best_dice:.4f}")
    print(f"Checkpoints saved in: {args.checkpoint_dir}")
    print("="*60)


if __name__ == '__main__':
    main()
