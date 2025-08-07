#!/usr/bin/env python3
"""
RetouchFormer Single Inference Script
Does everything in one go:
1. Runs model inference
2. Creates fat comparison images (Input|Output|Target)
3. Computes and prints metrics
4. Saves CSV metrics in the same directory
"""

import os
import glob
import argparse
import importlib
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from core.dataset import wildDataset
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
import pandas as pd


def load_model(model_name="RetouchFormer", checkpoint_path="release_model", epoch="best"):
    """Load the RetouchFormer model"""
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    
    # CRITICAL: Set CUDA device BEFORE importing model to avoid custom op conflicts
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    
    net = importlib.import_module('model.' + model_name)
    model = net.InpaintGenerator().to(device)
    
    model_path = f"{checkpoint_path}/gen_{epoch}.pth"
    data = torch.load(model_path, map_location=device)
    model.load_state_dict(data)
    print(f'Loading model from: {model_path}')
    model.eval()
    return model, device


def run_inference(model, device, input_path, temp_output_dir):
    """Run inference on images"""
    test_dataset = wildDataset(input_path)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    os.makedirs(temp_output_dir, exist_ok=True)
    
    print(f"Running inference on {len(test_dataset)} images...")
    for name, source_tensor in tqdm(test_loader, desc="Inference"):
        name = name[0]
        with torch.no_grad():
            pred_img, _ = model(source_tensor.to(device))
            path = os.path.join(temp_output_dir, f"{str(name)}_out.png")
            save_image(pred_img, path, normalize=True, value_range=(-1, 1))


def load_and_resize_image(image_path, size=512):
    """Load and resize image to specified size"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.LANCZOS)
    return img


def compute_metrics(img1, img2):
    """Compute image quality metrics between two images"""
    img1_arr = np.array(img1)
    img2_arr = np.array(img2)
    
    # Convert to float for calculations
    img1_f = img1_arr.astype(np.float64)
    img2_f = img2_arr.astype(np.float64)
    
    # Compute metrics
    mse_value = mse(img1_f, img2_f)
    mae_value = np.mean(np.abs(img1_f - img2_f))
    psnr_value = psnr(img1_arr, img2_arr, data_range=255)
    ssim_value = ssim(img1_arr, img2_arr, multichannel=True, channel_axis=2, data_range=255)
    
    return {
        'MSE': mse_value,
        'MAE': mae_value,
        'PSNR': psnr_value,
        'SSIM': ssim_value
    }


def create_fat_images_and_compute_metrics(source_dir, target_dir, temp_output_dir, final_output_dir):
    """Create fat comparison images and compute metrics"""
    os.makedirs(final_output_dir, exist_ok=True)
    
    source_images = sorted(glob.glob(os.path.join(source_dir, "*.png")))
    results = []
    
    print(f"Creating fat images and computing metrics for {len(source_images)} samples...")
    
    for source_path in tqdm(source_images, desc="Processing"):
        base_name = os.path.splitext(os.path.basename(source_path))[0]
        
        # Paths
        output_path = os.path.join(temp_output_dir, f"{base_name}_out.png")
        target_path = os.path.join(target_dir, f"{base_name}.png")
        fat_image_path = os.path.join(final_output_dir, f"{base_name}_comparison.png")
        
        # Check if files exist
        if not os.path.exists(output_path):
            print(f"Output file not found: {output_path}")
            continue
        if not os.path.exists(target_path):
            print(f"Target file not found: {target_path}")
            continue
        
        # Load and resize images
        source_img = load_and_resize_image(source_path)
        output_img = load_and_resize_image(output_path)
        target_img = load_and_resize_image(target_path)
        
        # Create fat comparison image (Input | Output | Target)
        width, height = source_img.size
        fat_image = Image.new('RGB', (width * 3, height))
        fat_image.paste(source_img, (0, 0))
        fat_image.paste(output_img, (width, 0))
        fat_image.paste(target_img, (width * 2, 0))
        fat_image.save(fat_image_path)
        
        # Compute metrics
        input_vs_target = compute_metrics(source_img, target_img)
        output_vs_target = compute_metrics(output_img, target_img)
        
        # Store results
        result = {
            'Image': base_name,
            'Input_vs_Target_MSE': input_vs_target['MSE'],
            'Input_vs_Target_MAE': input_vs_target['MAE'],
            'Input_vs_Target_PSNR': input_vs_target['PSNR'],
            'Input_vs_Target_SSIM': input_vs_target['SSIM'],
            'Output_vs_Target_MSE': output_vs_target['MSE'],
            'Output_vs_Target_MAE': output_vs_target['MAE'],
            'Output_vs_Target_PSNR': output_vs_target['PSNR'],
            'Output_vs_Target_SSIM': output_vs_target['SSIM'],
            'MSE_Improvement': input_vs_target['MSE'] - output_vs_target['MSE'],
            'MAE_Improvement': input_vs_target['MAE'] - output_vs_target['MAE'],
            'PSNR_Improvement': output_vs_target['PSNR'] - input_vs_target['PSNR'],
            'SSIM_Improvement': output_vs_target['SSIM'] - input_vs_target['SSIM'],
        }
        
        results.append(result)
    
    return results


def print_metrics_summary(results):
    """Print comprehensive metrics summary"""
    if not results:
        print("No results to display")
        return
    
    df = pd.DataFrame(results)
    
    print("RETOUCHFORMER METRICS SUMMARY")

    print(f"PROCESSED: {len(df)} images")
    
    print(f"AVERAGE METRICS:")
    print(f"{'Metric':<8} {'Input vs Target':<18} {'Output vs Target':<18} {'Improvement':<12}")

    metrics = ['MSE', 'MAE', 'PSNR', 'SSIM']
    for metric in metrics:
        input_col = f'Input_vs_Target_{metric}'
        output_col = f'Output_vs_Target_{metric}'
        improvement_col = f'{metric}_Improvement'
        
        input_avg = df[input_col].mean()
        output_avg = df[output_col].mean()
        improvement_avg = df[improvement_col].mean()
        
        print(f"{metric:<8} {input_avg:<18.4f} {output_avg:<18.4f} {improvement_avg:<12.4f}")
    
    print(f"IMPROVEMENT ANALYSIS:")
    mse_improvements = (df['MSE_Improvement'] > 0).sum()
    mae_improvements = (df['MAE_Improvement'] > 0).sum()
    psnr_improvements = (df['PSNR_Improvement'] > 0).sum()
    ssim_improvements = (df['SSIM_Improvement'] > 0).sum()
    
    total = len(df)
    print(f"Images with MSE improvement:  {mse_improvements}/{total} ({mse_improvements/total*100:.1f}%)")
    print(f"Images with MAE improvement:  {mae_improvements}/{total} ({mae_improvements/total*100:.1f}%)")
    print(f"Images with PSNR improvement: {psnr_improvements}/{total} ({psnr_improvements/total*100:.1f}%)")
    print(f"Images with SSIM improvement: {ssim_improvements}/{total} ({ssim_improvements/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="RetouchFormer Single Inference Script")
    parser.add_argument("--source_dir", type=str,
                       default="/data/data/face_retouch/face_retouching_subset/train/source",
                       help="Path to source images directory")
    parser.add_argument("--target_dir", type=str,
                       default="/data/data/face_retouch/face_retouching_subset/train/target",
                       help="Path to target images directory")
    parser.add_argument("--output_dir", type=str, default="retouchformer_results",
                       help="Directory to save fat images and metrics")
    parser.add_argument("--model", type=str, default="RetouchFormer",
                       help="Model name")
    parser.add_argument("--checkpoint_path", type=str, default="release_model",
                       help="Path to model checkpoint directory")
    parser.add_argument("--epoch", type=str, default="best",
                       help="Checkpoint epoch to load")
    
    args = parser.parse_args()
    
    print("RetouchFormer Single Inference Script")

    # Create temporary directory for inference outputs
    temp_output_dir = "temp_inference_outputs"
    
    try:
        print("Loading model")
        model, device = load_model(args.model, args.checkpoint_path, args.epoch)
        
        # Step 2: Run inference
        print("Running inference")
        run_inference(model, device, args.source_dir, temp_output_dir)
        
        # Step 3: Create fat images and compute metrics
        print("Creating fat images and computing metrics")
        results = create_fat_images_and_compute_metrics(
            args.source_dir, args.target_dir, temp_output_dir, args.output_dir
        )
        
        # Step 4: Save metrics CSV
        if results:
            csv_path = os.path.join(args.output_dir, "metrics.csv")
            df = pd.DataFrame(results)
            df.to_csv(csv_path, index=False)
            print(f"Metrics saved to: {csv_path}")
        
        # Step 5: Print summary
        print_metrics_summary(results)
        
        print("INFERENCE COMPLETE!")
        print(f"Fat comparison images: {args.output_dir}/")
        print(f"Metrics CSV: {args.output_dir}/metrics.csv")
        print(f"Total images processed: {len(results)}")
        
    finally:
        # Cleanup temporary directory
        if os.path.exists(temp_output_dir):
            import shutil
            shutil.rmtree(temp_output_dir)
            print(f"Cleaned up temporary files")


if __name__ == "__main__":
    main() 