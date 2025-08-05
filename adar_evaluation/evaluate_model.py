#!/usr/bin/env python3
import os
import sys
import json
import argparse
import glob
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
import importlib
from PIL import Image
import cv2

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dataset import FaceRetouchingDataset

def load_config_from_checkpoint(checkpoint_dir):
    """Load config from checkpoint directory"""
    config_files = glob.glob(os.path.join(checkpoint_dir, "*.json"))
    if not config_files:
        raise FileNotFoundError(f"No config file found in {checkpoint_dir}")
    
    config_path = config_files[0]
    print(f"📋 Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def find_latest_checkpoint(checkpoint_dir):
    """Find latest checkpoint in directory"""
    # Check for latest.ckpt file and look for generator checkpoint
    latest_file = os.path.join(checkpoint_dir, 'latest.ckpt')
    if os.path.isfile(latest_file):
        with open(latest_file, 'r') as f:
            latest_epoch = f.read().strip()
        # Look for generator checkpoint first
        gen_checkpoint_path = os.path.join(checkpoint_dir, f'gen_{latest_epoch}.pth')
        if os.path.exists(gen_checkpoint_path):
            return gen_checkpoint_path, latest_epoch
        # Fallback to regular checkpoint name
        checkpoint_path = os.path.join(checkpoint_dir, f'{latest_epoch}.pth')
        if os.path.exists(checkpoint_path):
            return checkpoint_path, latest_epoch
    
    # Fallback: find latest generator .pth file
    gen_ckpts = glob.glob(os.path.join(checkpoint_dir, 'gen_*.pth'))
    if gen_ckpts:
        # Sort by the numeric part of the filename
        gen_ckpts.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]), reverse=True)
        latest_checkpoint = gen_ckpts[0]
        epoch = os.path.basename(latest_checkpoint).split('_')[1].split('.pth')[0]
        return latest_checkpoint, epoch
    
    # Last resort: find any .pth file
    ckpts = glob.glob(os.path.join(checkpoint_dir, '*.pth'))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    
    # Sort by modification time
    ckpts.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_checkpoint = ckpts[0]
    epoch = os.path.basename(latest_checkpoint).split('.pth')[0]
    
    return latest_checkpoint, epoch

def load_model(checkpoint_path, config, device):
    """Load model from checkpoint"""
    print(f"🔄 Loading model from: {checkpoint_path}")
    
    # Import and create model
    net = importlib.import_module('model.' + config['model']['net'])
    model = net.InpaintGenerator()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'netG' in checkpoint:
        state_dict = checkpoint['netG']
    elif 'generator' in checkpoint:
        state_dict = checkpoint['generator']
    elif 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        # If checkpoint is the state dict itself, use it directly
        # Filter out non-model keys (training metadata)
        state_dict = {}
        for key, value in checkpoint.items():
            # Skip training metadata keys
            if key not in ['epoch', 'iteration', 'optimG', 'optim_maskG', 'optimD', 'scheG', 'sche_maskG', 'scheD']:
                state_dict[key] = value
    
    # Remove module. prefix if present (from DDP training)
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove 'module.' prefix
        else:
            new_key = key
        new_state_dict[new_key] = value
    
    model.load_state_dict(new_state_dict)
    model = model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
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
                if isinstance(name, (list, tuple)):
                    name = name[0]
            else:
                source_tensor, target_tensor = batch
                name = f"test_{batch_idx:04d}"
            
            source_tensor = source_tensor.to(device)
            target_tensor = target_tensor.to(device)
            
            # Generate prediction
            pred_tensor, _ = model(source_tensor)
            
            # Convert tensors to numpy for metrics computation
            pred_np = pred_tensor[0].cpu().numpy()
            target_np = target_tensor[0].cpu().numpy()
            source_np = source_tensor[0].cpu().numpy()
            
            # Ensure proper range [0, 1] for metrics computation
            pred_np = np.clip((pred_np + 1.0) / 2.0, 0.0, 1.0)
            target_np = np.clip((target_np + 1.0) / 2.0, 0.0, 1.0)
            source_np = np.clip((source_np + 1.0) / 2.0, 0.0, 1.0)
            
            # Compute metrics
            psnr_val = compare_psnr(target_np, pred_np, data_range=1.0)
            ssim_val = compare_ssim(target_np, pred_np, channel_axis=0, data_range=1.0)
            lpips_val = lpips_fn(pred_tensor, target_tensor).item()
            l1_val = F.l1_loss(pred_tensor, target_tensor).item()
            l2_val = F.mse_loss(pred_tensor, target_tensor).item()
            
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
                
                # Save individual images
                cv2.imwrite(os.path.join(output_dir, f"{name}_comparison.jpg"), 
                          cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(output_dir, f"{name}_prediction.jpg"), 
                          cv2.cvtColor(pred_img, cv2.COLOR_RGB2BGR))
    
    # Compute final statistics
    results = {
        'mean': {
            'PSNR': cumulative_psnr / total_samples,
            'SSIM': cumulative_ssim / total_samples,
            'LPIPS': cumulative_lpips / total_samples,
            'L1': cumulative_l1 / total_samples,
            'L2': cumulative_l2 / total_samples
        },
        'std': {
            'PSNR': np.std(psnr_values),
            'SSIM': np.std(ssim_values),
            'LPIPS': np.std(lpips_values),
            'L1': np.std(l1_values),
            'L2': np.std(l2_values)
        },
        'min': {
            'PSNR': np.min(psnr_values),
            'SSIM': np.min(ssim_values),
            'LPIPS': np.min(lpips_values),
            'L1': np.min(l1_values),
            'L2': np.min(l2_values)
        },
        'max': {
            'PSNR': np.max(psnr_values),
            'SSIM': np.max(ssim_values),
            'LPIPS': np.max(lpips_values),
            'L1': np.max(l1_values),
            'L2': np.max(l2_values)
        },
        'total_samples': total_samples
    }
    
    return results

def print_results(results, checkpoint_path, epoch):
    """Print formatted evaluation results"""
    print("\n" + "="*80)
    print(f"📊 EVALUATION RESULTS")
    print("="*80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch/Iteration: {epoch}")
    print(f"Total samples: {results['total_samples']}")
    print("-"*80)
    print(f"{'Metric':<12} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*80)
    
    for metric in ['PSNR', 'SSIM', 'LPIPS', 'L1', 'L2']:
        print(f"{metric:<12} "
              f"{results['mean'][metric]:<12.4f} "
              f"{results['std'][metric]:<12.4f} "
              f"{results['min'][metric]:<12.4f} "
              f"{results['max'][metric]:<12.4f}")
    
    print("-"*80)
    print("📈 Quality Assessment:")
    
    # Quality assessment
    mean_psnr = results['mean']['PSNR']
    mean_ssim = results['mean']['SSIM']
    mean_lpips = results['mean']['LPIPS']
    
    print(f"  PSNR: {mean_psnr:.2f} dB", end="")
    if mean_psnr > 25:
        print(" - ✅ Excellent")
    elif mean_psnr > 20:
        print(" - 👍 Good")
    elif mean_psnr > 15:
        print(" - ⚠️  Fair")
    else:
        print(" - ❌ Poor")
    
    print(f"  SSIM: {mean_ssim:.3f}", end="")
    if mean_ssim > 0.85:
        print(" - ✅ Excellent")
    elif mean_ssim > 0.75:
        print(" - 👍 Good")
    elif mean_ssim > 0.65:
        print(" - ⚠️  Fair")
    else:
        print(" - ❌ Poor")
    
    print(f"  LPIPS: {mean_lpips:.3f}", end="")
    if mean_lpips < 0.15:
        print(" - ✅ Excellent")
    elif mean_lpips < 0.25:
        print(" - 👍 Good")
    elif mean_lpips < 0.4:
        print(" - ⚠️  Fair")
    else:
        print(" - ❌ Poor")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Evaluate RetouchFormer Model')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint and config files')
    parser.add_argument('--checkpoint_file', type=str, default=None,
                       help='Specific checkpoint file to evaluate (optional, will use latest if not specified)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run evaluation on')
    parser.add_argument('--save_images', action='store_true',
                       help='Save comparison images')
    parser.add_argument('--output_dir', type=str, default='adar_evaluation/evaluation_results',
                       help='Directory to save evaluation results and images')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    print("🎯 RetouchFormer Model Evaluation")
    print("="*60)
    
    # Setup device
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
        print(f"🚀 Using device: {device}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # Load config
    config = load_config_from_checkpoint(args.checkpoint_dir)
    
    # Find checkpoint
    if args.checkpoint_file:
        checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
        epoch = os.path.basename(args.checkpoint_file).split('.pth')[0]
    else:
        checkpoint_path, epoch = find_latest_checkpoint(args.checkpoint_dir)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    # Load model
    model = load_model(checkpoint_path, config, device)
    
    # Setup test dataset
    print(f"📂 Loading test dataset from: {config['train_data_loader']['dataroot']}")
    test_dataset = FaceRetouchingDataset(
        path=config['train_data_loader']['dataroot'],
        resolution=config['train_data_loader']['size'],
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
    print_results(results, checkpoint_path, epoch)
    
    # Save results to file
    output_file = os.path.join(args.output_dir, 'evaluation_metrics.json')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Add metadata to results
    results['metadata'] = {
        'checkpoint_path': checkpoint_path,
        'epoch': epoch,
        'config': config,
        'device': str(device)
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    if args.save_images:
        print(f"🖼️  Images saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 