"""
Testing/Evaluation Script for MK-GAN
"""

import torch
import argparse
import os
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from scipy.spatial.distance import directed_hausdorff
from scipy.ndimage import distance_transform_edt

from MKGAN import MKGANGenerator
from dataloader import get_dataloader


# ============================================================================
# Metrics
# ============================================================================

def calculate_dice(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Dice Similarity Coefficient"""
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > threshold).astype(np.float32)

    intersection = np.sum(pred_bin * target_bin)
    union = np.sum(pred_bin) + np.sum(target_bin)

    if union == 0:
        return 1.0 if np.sum(target_bin) == 0 else 0.0

    return (2.0 * intersection) / (union + 1e-8)


def calculate_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Intersection over Union"""
    pred_bin = (pred > threshold).astype(np.float32)
    target_bin = (target > threshold).astype(np.float32)

    intersection = np.sum(pred_bin * target_bin)
    union = np.sum(pred_bin) + np.sum(target_bin) - intersection

    if union == 0:
        return 1.0 if np.sum(target_bin) == 0 else 0.0

    return intersection / (union + 1e-8)


def calculate_hausdorff(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Hausdorff Distance"""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)

    pred_points = np.argwhere(pred_bin > 0)
    target_points = np.argwhere(target_bin > 0)

    if len(pred_points) == 0 or len(target_points) == 0:
        return 0.0

    forward = directed_hausdorff(pred_points, target_points)[0]
    backward = directed_hausdorff(target_points, pred_points)[0]

    return max(forward, backward)


def calculate_mad(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """Mean Absolute Distance"""
    pred_bin = (pred > threshold).astype(np.uint8)
    target_bin = (target > threshold).astype(np.uint8)

    if pred_bin.sum() == 0 or target_bin.sum() == 0:
        return 0.0

    # Distance transforms
    pred_dist = distance_transform_edt(1 - pred_bin)
    target_dist = distance_transform_edt(1 - target_bin)

    # Boundaries
    pred_boundary = pred_bin - cv2.erode(pred_bin, np.ones((3, 3), np.uint8))
    target_boundary = target_bin - cv2.erode(target_bin, np.ones((3, 3), np.uint8))

    # Average distances
    d1 = pred_boundary * target_dist
    d2 = target_boundary * pred_dist

    n1 = pred_boundary.sum()
    n2 = target_boundary.sum()

    if n1 == 0 or n2 == 0:
        return 0.0

    mad = (d1.sum() / n1 + d2.sum() / n2) / 2
    return mad


def calculate_all_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    """Calculate all segmentation metrics"""
    return {
        'dice': calculate_dice(pred, target),
        'iou': calculate_iou(pred, target),
        'hd': calculate_hausdorff(pred, target),
        'mad': calculate_mad(pred, target)
    }


# ============================================================================
# Visualization
# ============================================================================

def save_visualization(image: np.ndarray, gt: np.ndarray, pred: np.ndarray,
                       save_path: str, metrics: dict = None):
    """Save side-by-side visualization"""
    h, w = image.shape[:2]

    # Convert to 3-channel if grayscale
    if len(image.shape) == 2:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_rgb = image.copy()

    # Binarize masks
    gt_bin = (gt > 0.5).astype(np.uint8) * 255
    pred_bin = (pred > 0.5).astype(np.uint8) * 255

    # Create overlay
    overlay = image_rgb.copy()
    # GT in red
    overlay[gt_bin > 0] = [0, 0, 255]
    # Prediction in green (will overlap)
    pred_only = (pred_bin > 0) & (gt_bin == 0)
    overlay[pred_only] = [0, 255, 0]

    # Convert masks to BGR for stacking
    gt_bgr = cv2.cvtColor(gt_bin, cv2.COLOR_GRAY2BGR)
    pred_bgr = cv2.cvtColor(pred_bin, cv2.COLOR_GRAY2BGR)

    # Stack horizontally
    combined = np.hstack([image_rgb, gt_bgr, pred_bgr, overlay])

    # Add text with metrics
    if metrics:
        text = f"Dice: {metrics['dice']:.3f} | IoU: {metrics['iou']:.3f} | HD: {metrics['hd']:.2f}"
        cv2.putText(combined, text, (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite(str(save_path), combined)


# ============================================================================
# Testing Function
# ============================================================================

def test(generator, dataloader, device, save_dir=None, save_max=50):
    """Evaluate the model on test set"""
    generator.eval()

    all_metrics = {'dice': [], 'iou': [], 'hd': [], 'mad': []}

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(dataloader, desc='Testing')

    with torch.no_grad():
        for idx, (images, masks) in enumerate(pbar):
            images = images.to(device)
            masks = masks.to(device)

            outputs, _ = generator(images)

            # Convert to numpy (use first channel for binary segmentation)
            img_np = images[0, 0].cpu().numpy()
            img_np = ((img_np + 1) / 2 * 255).astype(np.uint8)  # Denormalize

            mask_np = masks[0, 0].cpu().numpy()
            pred_np = outputs[0, 0].cpu().numpy()

            # Calculate metrics
            metrics = calculate_all_metrics(pred_np, mask_np)

            for key in all_metrics:
                all_metrics[key].append(metrics[key])

            # Save visualization for first N samples
            if save_dir and idx < save_max:
                save_path = save_dir / f'result_{idx:04d}.png'
                save_visualization(img_np, mask_np, pred_np, save_path, metrics)

            pbar.set_postfix({
                'Dice': f"{metrics['dice']:.3f}",
                'IoU': f"{metrics['iou']:.3f}"
            })

    # Aggregate results
    results = {}
    for key in all_metrics:
        values = np.array(all_metrics[key])
        results[f'{key}_mean'] = float(values.mean())
        results[f'{key}_std'] = float(values.std())
        results[f'{key}_median'] = float(np.median(values))

    return results, all_metrics


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Test MK-GAN')

    parser.add_argument('--test_image_dir', type=str, required=True,
                        help='Test images directory')
    parser.add_argument('--test_mask_dir', type=str, required=True,
                        help='Test masks directory')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Model arguments
    parser.add_argument('--in_channels', type=int, default=3)
    parser.add_argument('--out_channels', type=int, default=3)
    parser.add_argument('--base_filters', type=int, default=48)
    parser.add_argument('--img_size', type=int, default=256)

    # Output
    parser.add_argument('--save_dir', type=str, default='./results',
                        help='Directory to save visualizations')
    parser.add_argument('--save_max', type=int, default=50,
                        help='Maximum number of visualizations to save')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Create and load model
    print("\nLoading model...")
    generator = MKGANGenerator(
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        base_filters=args.base_filters
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    generator.load_state_dict(checkpoint['generator'])

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Best validation Dice: {checkpoint.get('best_dice', 'N/A')}")

    # Data loader
    print("\nLoading test data...")
    test_loader = get_dataloader(
        image_dir=args.test_image_dir,
        mask_dir=args.test_mask_dir,
        batch_size=1,
        img_size=args.img_size,
        num_workers=args.num_workers,
        mode='test',
        shuffle=False
    )

    # Run testing
    print("\n" + "="*60)
    print("Running evaluation...")
    print("="*60)

    results, raw_metrics = test(
        generator, test_loader, device,
        save_dir=args.save_dir,
        save_max=args.save_max
    )

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Median':<12}")
    print("-"*60)
    for metric in ['dice', 'iou', 'hd', 'mad']:
        mean_val = results[f'{metric}_mean']
        std_val = results[f'{metric}_std']
        median_val = results[f'{metric}_median']
        unit = ' mm' if metric in ['hd', 'mad'] else ''
        print(f"{metric.upper():<15} {mean_val:.4f}{unit:<6} "
              f"{std_val:.4f}{unit:<6} {median_val:.4f}{unit}")
    print("="*60)

    # Save results to file
    if args.save_dir:
        results_file = Path(args.save_dir) / 'results.txt'
        with open(results_file, 'w') as f:
            f.write("MK-GAN Evaluation Results\n")
            f.write("="*60 + "\n")
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Test set: {args.test_image_dir}\n")
            f.write(f"Number of samples: {len(test_loader)}\n\n")
            f.write(f"{'Metric':<15} {'Mean':<12} {'Std':<12} {'Median':<12}\n")
            f.write("-"*60 + "\n")
            for metric in ['dice', 'iou', 'hd', 'mad']:
                f.write(f"{metric.upper():<15} {results[f'{metric}_mean']:.4f}     "
                        f"{results[f'{metric}_std']:.4f}     "
                        f"{results[f'{metric}_median']:.4f}\n")
        print(f"\nResults saved to: {results_file}")
        print(f"Visualizations saved to: {args.save_dir}")


if __name__ == '__main__':
    main()
