#!/usr/bin/env python3
import json
import argparse

def load_results(result_file):
    """Load results from JSON file"""
    with open(result_file, 'r') as f:
        return json.load(f)

def get_metric_value(results, metric_type, metric_name):
    """Get metric value handling different JSON structures"""
    # Handle different key formats
    if f"{metric_type}_metrics" in results:
        value = results[f"{metric_type}_metrics"][metric_name]
    elif metric_type in results:
        value = results[metric_type][metric_name]
    else:
        raise KeyError(f"Cannot find {metric_type} for {metric_name}")
    
    # Convert string values to float if needed
    if isinstance(value, str):
        value = float(value)
    
    return value

def print_comparison(results_list, model_names):
    """Print side-by-side comparison of results"""
    print("\n" + "="*100)
    print("🏆 MODEL COMPARISON RESULTS")
    print("="*100)
    
    # Header
    print(f"{'Metric':<12}", end="")
    for name in model_names:
        print(f"{name[:20]:<22}", end="")
    print()
    print("-" * 100)
    
    # Metrics comparison
    metrics = ['PSNR', 'SSIM', 'LPIPS', 'L1', 'L2']
    
    for metric in metrics:
        print(f"{metric:<12}", end="")
        for results in results_list:
            try:
                mean_val = get_metric_value(results, 'mean', metric)
                std_val = get_metric_value(results, 'std', metric)
                
                if metric == 'PSNR':
                    print(f"{mean_val:6.2f} ± {std_val:5.2f} dB     ", end="")
                elif metric in ['SSIM', 'LPIPS']:
                    print(f"{mean_val:6.3f} ± {std_val:6.3f}      ", end="")
                else:
                    print(f"{mean_val:6.4f} ± {std_val:6.4f}      ", end="")
            except (KeyError, ValueError) as e:
                print(f"{'N/A':<22}", end="")
        print()
    
    print("-" * 100)
    
    # Winner analysis
    print("🎯 WINNER ANALYSIS:")
    for metric in ['PSNR', 'SSIM', 'LPIPS']:
        print(f"\n{metric}:")
        values = []
        valid_indices = []
        
        for i, results in enumerate(results_list):
            try:
                value = get_metric_value(results, 'mean', metric)
                values.append(value)
                valid_indices.append(i)
            except (KeyError, ValueError):
                continue
        
        if not values:
            print(f"  No valid data for {metric}")
            continue
            
        if metric == 'LPIPS':  # Lower is better for LPIPS
            winner_idx = valid_indices[values.index(min(values))]
        else:  # Higher is better for PSNR and SSIM
            winner_idx = valid_indices[values.index(max(values))]
        
        for i, (results, name) in enumerate(zip(results_list, model_names)):
            try:
                value = get_metric_value(results, 'mean', metric)
                if i == winner_idx:
                    if metric == 'PSNR':
                        print(f"  🥇 {name}: {value:.2f} dB")
                    else:
                        print(f"  🥇 {name}: {value:.3f}")
                else:
                    if metric == 'PSNR':
                        print(f"  🥈 {name}: {value:.2f} dB")
                    else:
                        print(f"  🥈 {name}: {value:.3f}")
            except (KeyError, ValueError):
                print(f"  ❓ {name}: N/A")
    
    print("\n" + "="*100)
    
    # Performance differences
    if len(results_list) == 2:
        print("📊 PERFORMANCE DIFFERENCE:")
        authors_results = results_list[0]
        enhanced_results = results_list[1]
        
        try:
            psnr_authors = get_metric_value(authors_results, 'mean', 'PSNR')
            psnr_enhanced = get_metric_value(enhanced_results, 'mean', 'PSNR')
            psnr_diff = psnr_enhanced - psnr_authors
            
            ssim_authors = get_metric_value(authors_results, 'mean', 'SSIM')
            ssim_enhanced = get_metric_value(enhanced_results, 'mean', 'SSIM')
            ssim_diff = ssim_enhanced - ssim_authors
            
            lpips_authors = get_metric_value(authors_results, 'mean', 'LPIPS')
            lpips_enhanced = get_metric_value(enhanced_results, 'mean', 'LPIPS')
            lpips_diff = lpips_enhanced - lpips_authors
            
            print(f"PSNR difference: {psnr_diff:+.2f} dB")
            print(f"SSIM difference: {ssim_diff:+.4f}")
            print(f"LPIPS difference: {lpips_diff:+.4f} (lower is better)")
            
            # Overall assessment
            print("\n🎯 OVERALL ASSESSMENT:")
            if abs(psnr_diff) < 1.0 and abs(ssim_diff) < 0.005 and abs(lpips_diff) < 0.005:
                print("✅ Enhanced model performs COMPETITIVELY with authors' model")
                print("   The differences are minimal and both models show excellent performance")
            elif psnr_diff > 0 and ssim_diff > 0 and lpips_diff < 0:
                print("🚀 Enhanced model OUTPERFORMS authors' model!")
            else:
                print("📈 Enhanced model shows good performance but slightly behind authors' model")
                print("   This is expected as the authors' model is the published result")
        except (KeyError, ValueError) as e:
            print(f"❌ Could not compute performance differences: {e}")

def main():
    parser = argparse.ArgumentParser(description='Compare Model Evaluation Results')
    parser.add_argument('result_files', nargs='+', help='JSON result files to compare')
    parser.add_argument('--model_names', nargs='*', help='Names for the models')
    
    args = parser.parse_args()
    
    # Load results
    results_list = []
    for file_path in args.result_files:
        results = load_results(file_path)
        results_list.append(results)
    
    # Generate names if not provided
    if args.model_names and len(args.model_names) == len(results_list):
        model_names = args.model_names
    else:
        model_names = [f"Model_{i+1}" for i in range(len(results_list))]
    
    # Print comparison
    print_comparison(results_list, model_names)

if __name__ == "__main__":
    main() 