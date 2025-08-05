#!/usr/bin/env python3
import os
import json
import argparse
import subprocess
import sys
from datetime import datetime
import pandas as pd

def run_evaluation(checkpoint_dir, device='cuda:0', output_dir=None, max_samples=None):
    """Run evaluation on a checkpoint directory"""
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"evaluation_{os.path.basename(checkpoint_dir)}_{timestamp}"
    
    print(f"\n🔄 Evaluating: {checkpoint_dir}")
    print(f"📁 Output directory: {output_dir}")
    
    cmd = [
        sys.executable, 'adar_evaluation/evaluate_model.py',
        '--checkpoint_dir', checkpoint_dir,
        '--device', device,
        '--output_dir', output_dir,
        '--save_images'
    ]
    
    if max_samples is not None:
        cmd.extend(['--max_samples', str(max_samples)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Evaluation completed successfully")
        
        # Load and return results
        results_file = os.path.join(output_dir, 'evaluation_metrics.json')
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                results = json.load(f)
            return results, output_dir
        else:
            print(f"⚠️  Results file not found: {results_file}")
            return None, output_dir
    except subprocess.CalledProcessError as e:
        print(f"❌ Evaluation failed: {e}")
        print(f"Error output: {e.stderr}")
        return None, output_dir

def create_comparison_table(results_list, model_names):
    """Create a comparison table from multiple evaluation results"""
    if not results_list or not any(results_list):
        print("❌ No valid results to compare")
        return None
    
    # Prepare data for comparison table
    comparison_data = []
    
    for i, (results, model_name) in enumerate(zip(results_list, model_names)):
        if results is None:
            print(f"⚠️  Skipping {model_name} - no results available")
            continue
            
        mean_metrics = results['mean']
        std_metrics = results['std']
        metadata = results.get('metadata', {})
        
        row = {
            'Model': model_name,
            'Checkpoint': os.path.basename(metadata.get('checkpoint_path', 'unknown')),
            'Epoch': metadata.get('epoch', 'unknown'),
            'Samples': results['total_samples'],
            'PSNR (dB)': f"{mean_metrics['PSNR']:.2f} ± {std_metrics['PSNR']:.2f}",
            'SSIM': f"{mean_metrics['SSIM']:.3f} ± {std_metrics['SSIM']:.3f}",
            'LPIPS': f"{mean_metrics['LPIPS']:.3f} ± {std_metrics['LPIPS']:.3f}",
            'L1': f"{mean_metrics['L1']:.4f} ± {std_metrics['L1']:.4f}",
            'L2': f"{mean_metrics['L2']:.4f} ± {std_metrics['L2']:.4f}",
            'PSNR_mean': mean_metrics['PSNR'],  # For sorting
            'SSIM_mean': mean_metrics['SSIM'],
            'LPIPS_mean': mean_metrics['LPIPS']
        }
        comparison_data.append(row)
    
    if not comparison_data:
        print("❌ No valid data for comparison")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(comparison_data)
    
    # Sort by PSNR (descending) then by SSIM (descending)
    df = df.sort_values(['PSNR_mean', 'SSIM_mean'], ascending=[False, False])
    
    # Remove sorting columns
    display_df = df.drop(['PSNR_mean', 'SSIM_mean', 'LPIPS_mean'], axis=1)
    
    return display_df, df

def print_comparison_results(df, df_full):
    """Print formatted comparison results"""
    print("\n" + "="*120)
    print("📊 MODEL COMPARISON RESULTS")
    print("="*120)
    
    # Print table
    print(df.to_string(index=False))
    
    print("\n" + "-"*120)
    print("🏆 RANKINGS:")
    print("-"*120)
    
    # Best PSNR
    best_psnr_idx = df_full['PSNR_mean'].idxmax()
    best_psnr = df_full.loc[best_psnr_idx]
    print(f"🥇 Best PSNR: {best_psnr['Model']} - {best_psnr['PSNR_mean']:.2f} dB")
    
    # Best SSIM
    best_ssim_idx = df_full['SSIM_mean'].idxmax()
    best_ssim = df_full.loc[best_ssim_idx]
    print(f"🥇 Best SSIM: {best_ssim['Model']} - {best_ssim['SSIM_mean']:.3f}")
    
    # Best LPIPS (lowest is better)
    best_lpips_idx = df_full['LPIPS_mean'].idxmin()
    best_lpips = df_full.loc[best_lpips_idx]
    print(f"🥇 Best LPIPS: {best_lpips['Model']} - {best_lpips['LPIPS_mean']:.3f}")
    
    print("\n" + "-"*120)
    print("📈 PERFORMANCE ANALYSIS:")
    print("-"*120)
    
    for _, row in df_full.iterrows():
        model_name = row['Model']
        psnr = row['PSNR_mean']
        ssim = row['SSIM_mean']
        lpips = row['LPIPS_mean']
        
        print(f"\n{model_name}:")
        
        # PSNR assessment
        print(f"  PSNR: {psnr:.2f} dB", end="")
        if psnr > 25:
            print(" - ✅ Excellent")
        elif psnr > 20:
            print(" - 👍 Good")
        elif psnr > 15:
            print(" - ⚠️  Fair")
        else:
            print(" - ❌ Poor")
        
        # SSIM assessment
        print(f"  SSIM: {ssim:.3f}", end="")
        if ssim > 0.85:
            print(" - ✅ Excellent")
        elif ssim > 0.75:
            print(" - 👍 Good")
        elif ssim > 0.65:
            print(" - ⚠️  Fair")
        else:
            print(" - ❌ Poor")
        
        # LPIPS assessment
        print(f"  LPIPS: {lpips:.3f}", end="")
        if lpips < 0.15:
            print(" - ✅ Excellent")
        elif lpips < 0.25:
            print(" - 👍 Good")
        elif lpips < 0.4:
            print(" - ⚠️  Fair")
        else:
            print(" - ❌ Poor")
    
    print("="*120)

def main():
    parser = argparse.ArgumentParser(description='Compare Multiple RetouchFormer Models')
    parser.add_argument('checkpoint_dirs', nargs='+', 
                       help='Directories containing checkpoints to compare')
    parser.add_argument('--model_names', nargs='*', default=None,
                       help='Custom names for the models (optional)')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run evaluations on')
    parser.add_argument('--output_dir', type=str, default='adar_evaluation/model_comparison',
                       help='Directory to save comparison results')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (None for all)')
    
    args = parser.parse_args()
    
    print("🔍 RetouchFormer Model Comparison Tool")
    print("="*60)
    
    # Validate checkpoint directories
    valid_dirs = []
    for checkpoint_dir in args.checkpoint_dirs:
        if not os.path.isdir(checkpoint_dir):
            print(f"⚠️  Skipping invalid directory: {checkpoint_dir}")
            continue
        valid_dirs.append(checkpoint_dir)
    
    if not valid_dirs:
        print("❌ No valid checkpoint directories provided")
        return
    
    # Generate model names if not provided
    if args.model_names:
        if len(args.model_names) != len(valid_dirs):
            print("⚠️  Number of model names doesn't match number of directories, using default names")
            model_names = [os.path.basename(d) for d in valid_dirs]
        else:
            model_names = args.model_names
    else:
        model_names = [os.path.basename(d) for d in valid_dirs]
    
    print(f"📋 Comparing {len(valid_dirs)} models:")
    for i, (dir_path, name) in enumerate(zip(valid_dirs, model_names), 1):
        print(f"  {i}. {name}: {dir_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run evaluations
    results_list = []
    output_dirs = []
    
    for checkpoint_dir, model_name in zip(valid_dirs, model_names):
        eval_output_dir = os.path.join(args.output_dir, f"eval_{model_name}")
        results, output_dir = run_evaluation(checkpoint_dir, args.device, eval_output_dir, args.max_samples)
        results_list.append(results)
        output_dirs.append(output_dir)
    
    # Create comparison table
    comparison_data = create_comparison_table(results_list, model_names)
    if comparison_data is None:
        print("❌ Failed to create comparison table")
        return
    
    display_df, full_df = comparison_data
    
    # Print results
    print_comparison_results(display_df, full_df)
    
    # Save comparison results
    comparison_file = os.path.join(args.output_dir, 'model_comparison.csv')
    display_df.to_csv(comparison_file, index=False)
    print(f"\n💾 Comparison table saved to: {comparison_file}")
    
    # Create summary report
    summary_file = os.path.join(args.output_dir, 'comparison_summary.txt')
    with open(summary_file, 'w') as f:
        f.write("RetouchFormer Model Comparison Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Models compared: {len(valid_dirs)}\n\n")
        
        f.write("Model Rankings:\n")
        f.write("-"*20 + "\n")
        
        best_psnr_idx = full_df['PSNR_mean'].idxmax()
        best_psnr = full_df.loc[best_psnr_idx]
        f.write(f"Best PSNR: {best_psnr['Model']} - {best_psnr['PSNR_mean']:.2f} dB\n")
        
        best_ssim_idx = full_df['SSIM_mean'].idxmax()
        best_ssim = full_df.loc[best_ssim_idx]
        f.write(f"Best SSIM: {best_ssim['Model']} - {best_ssim['SSIM_mean']:.3f}\n")
        
        best_lpips_idx = full_df['LPIPS_mean'].idxmin()
        best_lpips = full_df.loc[best_lpips_idx]
        f.write(f"Best LPIPS: {best_lpips['Model']} - {best_lpips['LPIPS_mean']:.3f}\n")
    
    print(f"📄 Summary report saved to: {summary_file}")
    print(f"📁 All evaluation results saved in: {args.output_dir}")

if __name__ == "__main__":
    main() 