#!/usr/bin/env python3
import os
import sys
import json
import argparse
import importlib
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from PIL import Image
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dataset import FaceRetouchingDataset

def load_authors_model(checkpoint_path, device):
    """Load authors' pre-trained model"""
    print(f"🔄 Loading authors' model from: {checkpoint_path}")
    
    # Import and create model - using RetouchFormer architecture
    net = importlib.import_module('model.RetouchFormer')
    model = net.InpaintGenerator()
    
    # Load checkpoint directly as state dict
    data = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(data)
    
    model = model.to(device)
    model.eval()
    
    print(f"✅ Authors' model loaded successfully")
    return model

def evaluate_model(model, test_loader, device, save_images=False, output_dir=None, max_samples=None):
    """Run comprehensive evaluation on test set"""
    total_available = len(test_loader.dataset)
    samples_to_eval = min(max_samples, total_available) if max_samples is not None else total_available
    print(f"🧪 Running evaluation on {samples_to_eval} test samples...")
    
    # Initialize metrics
    total_samples = 0
    cumulative_psnr = 0.0
    cumulative_ssim = 0.0
    cumulative_lpips = 0.0
    cumulative_l1 = 0.0
    cumulative_l2 = 0.0
    
    # Initialize LPIPS
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    # Store individual results for statistics
    psnr_values = []
    ssim_values = []
    lpips_values = []
    l1_values = []
    l2_values = []
    
    if save_images and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"💾 Saving images to: {output_dir}")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Check if we've reached max_samples limit
            if max_samples is not None and total_samples >= max_samples:
                break
            
            if len(batch) == 3:
                name, source_tensor, target_tensor = batch
            else:
                source_tensor, target_tensor = batch
                name = [f"sample_{batch_idx:05d}"]
            
            name = name[0] if isinstance(name, list) else name
            
            # Move to device
            source_tensor = source_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # Forward pass
            pred_tensor, _ = model(source_tensor)
            
            # Convert to numpy for metric computation
            pred_np = pred_tensor[0].cpu().numpy()
            target_np = target_tensor[0].cpu().numpy()
            source_np = source_tensor[0].cpu().numpy()
            
            # Normalize from [-1, 1] to [0, 1] for metrics
            pred_np = (pred_np + 1.0) / 2.0
            target_np = (target_np + 1.0) / 2.0
            source_np = (source_np + 1.0) / 2.0
            
            # Clip values to valid range
            pred_np = np.clip(pred_np, 0, 1)
            target_np = np.clip(target_np, 0, 1)
            source_np = np.clip(source_np, 0, 1)
            
            # Compute metrics
            psnr_val = compare_psnr(target_np, pred_np, data_range=1.0)
            ssim_val = compare_ssim(target_np, pred_np, channel_axis=0, data_range=1.0)
            
            # LPIPS expects [-1, 1] range
            lpips_val = lpips_fn(pred_tensor, target_tensor).item()
            
            # L1 and L2 losses on normalized tensors
            pred_tensor_norm = (pred_tensor + 1.0) / 2.0
            target_tensor_norm = (target_tensor + 1.0) / 2.0
            l1_val = F.l1_loss(pred_tensor_norm, target_tensor_norm).item()
            l2_val = F.mse_loss(pred_tensor_norm, target_tensor_norm).item()
            
            # Accumulate metrics
            cumulative_psnr += psnr_val
            cumulative_ssim += ssim_val
            cumulative_lpips += lpips_val
            cumulative_l1 += l1_val
            cumulative_l2 += l2_val
            total_samples += 1
            
            # Store individual values for statistics
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)
            lpips_values.append(lpips_val)
            l1_values.append(l1_val)
            l2_values.append(l2_val)
            
            # Save comparison images if requested
            if save_images and output_dir:
                # Convert to [0, 255] for saving
                pred_img = (pred_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                target_img = (target_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                source_img = (source_np.transpose(1, 2, 0) * 255).astype(np.uint8)
                
                # Create comparison image
                comparison = np.concatenate([source_img, pred_img, target_img], axis=1)
                
                # Save images
                cv2.imwrite(os.path.join(output_dir, f"{name}_comparison.jpg"), 
                          cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
    
    # Calculate final statistics
    mean_metrics = {
        'PSNR': cumulative_psnr / total_samples,
        'SSIM': cumulative_ssim / total_samples,
        'LPIPS': cumulative_lpips / total_samples,
        'L1': cumulative_l1 / total_samples,
        'L2': cumulative_l2 / total_samples
    }
    
    std_metrics = {
        'PSNR': np.std(psnr_values),
        'SSIM': np.std(ssim_values),
        'LPIPS': np.std(lpips_values),
        'L1': np.std(l1_values),
        'L2': np.std(l2_values)
    }
    
    min_metrics = {
        'PSNR': np.min(psnr_values),
        'SSIM': np.min(ssim_values),
        'LPIPS': np.min(lpips_values),
        'L1': np.min(l1_values),
        'L2': np.min(l2_values)
    }
    
    max_metrics = {
        'PSNR': np.max(psnr_values),
        'SSIM': np.max(ssim_values),
        'LPIPS': np.max(lpips_values),
        'L1': np.max(l1_values),
        'L2': np.max(l2_values)
    }
    
    return {
        'mean_metrics': mean_metrics,
        'std_metrics': std_metrics,
        'min_metrics': min_metrics,
        'max_metrics': max_metrics,
        'individual_values': {
            'PSNR': psnr_values,
            'SSIM': ssim_values,
            'LPIPS': lpips_values,
            'L1': l1_values,
            'L2': l2_values
        },
        'total_samples': total_samples
    }

def print_results(results, checkpoint_path):
    """Print formatted results"""
    mean_metrics = results['mean_metrics']
    std_metrics = results['std_metrics']
    min_metrics = results['min_metrics']
    max_metrics = results['max_metrics']
    
    print("\n" + "="*80)
    print("📊 EVALUATION RESULTS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: Authors' Pre-trained Model")
    print(f"Total samples: {results['total_samples']}")
    print("-" * 80)
    print(f"{'Metric':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 80)
    
    for metric in ['PSNR', 'SSIM', 'LPIPS', 'L1', 'L2']:
        print(f"{metric:<12} {mean_metrics[metric]:<12.4f} {std_metrics[metric]:<12.4f} "
              f"{min_metrics[metric]:<12.4f} {max_metrics[metric]:<12.4f}")
    
    print("-" * 80)
    print("📈 Quality Assessment:")
    
    # PSNR assessment
    psnr = mean_metrics['PSNR']
    print(f"  PSNR: {psnr:.2f} dB", end="")
    if psnr > 30:
        print(" - ✅ Excellent")
    elif psnr > 25:
        print(" - 👍 Good")
    elif psnr > 20:
        print(" - ⚠️  Fair")
    else:
        print(" - ❌ Poor")
    
    # SSIM assessment
    ssim = mean_metrics['SSIM']
    print(f"  SSIM: {ssim:.3f}", end="")
    if ssim > 0.95:
        print(" - ✅ Excellent")
    elif ssim > 0.85:
        print(" - 👍 Good")
    elif ssim > 0.75:
        print(" - ⚠️  Fair")
    else:
        print(" - ❌ Poor")
    
    # LPIPS assessment
    lpips_val = mean_metrics['LPIPS']
    print(f"  LPIPS: {lpips_val:.3f}", end="")
    if lpips_val < 0.15:
        print(" - ✅ Excellent")
    elif lpips_val < 0.25:
        print(" - 👍 Good")
    elif lpips_val < 0.4:
        print(" - ⚠️  Fair")
    else:
        print(" - ❌ Poor")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate Authors\' RetouchFormer Model')
    parser.add_argument('--checkpoint_path', type=str, default='release_model/gen_best.pth',
                       help='Path to authors\' checkpoint file')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run evaluation on')
    parser.add_argument('--save_images', action='store_true',
                       help='Save comparison images')
    parser.add_argument('--output_dir', type=str, default='adar_evaluation/evaluation_authors',
                       help='Directory to save evaluation results and images')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
    parser.add_argument('--dataset_path', type=str, 
                       default='/data/data/face_retouch/face_retouching_ffhqr_dataset',
                       help='Path to test dataset')
    
    args = parser.parse_args()
    
    print("🎯 Authors' RetouchFormer Model Evaluation")
    print("="*60)
    
    # Setup device
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
        print(f"🚀 Using device: {device}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint_path}")
    
    # Load model
    model = load_authors_model(args.checkpoint_path, device)
    
    # Setup test dataset
    print(f"📂 Loading test dataset from: {args.dataset_path}")
    test_dataset = FaceRetouchingDataset(
        path=args.dataset_path,
        resolution=512,
        data_type="test", 
        data_percentage=1
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )
    
    # Limit samples if max_samples is specified
    if args.max_samples is not None:
        print(f"📊 Test dataset loaded: {len(test_dataset)} samples (limiting to {args.max_samples})")
    else:
        print(f"📊 Test dataset loaded: {len(test_dataset)} samples")
    
    # Run evaluation
    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        save_images=args.save_images,
        output_dir=args.output_dir if args.save_images else None,
        max_samples=args.max_samples
    )
    
    # Print results
    print_results(results, args.checkpoint_path)
    
    # Save results to file
    output_file = os.path.join(args.output_dir, 'evaluation_metrics.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add metadata to results
    results['metadata'] = {
        'checkpoint_path': args.checkpoint_path,
        'model_type': 'authors_pretrained',
        'device': str(device)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    if args.save_images:
        print(f"🖼️  Images saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 