#!/usr/bin/env python3
"""
Test script to verify training setup before starting actual training
"""

import json
import torch
from core.trainer_enhanced import EnhancedTrainer
from core.dist import get_world_size, get_local_rank, get_global_rank, get_master_ip

def test_training_setup():
    print("🧪 Testing RetouchFormer Training Setup...")
    print("="*60)
    
    # Load config
    config_path = './configs/RetouchFormer_subset_trainer.json'
    config = json.load(open(config_path))
    
    # Set up minimal config for testing
    config['world_size'] = 1  # Force single GPU for testing
    config['init_method'] = f"tcp://{get_master_ip()}:2345"
    config['distributed'] = False  # Disable distributed for testing
    config['local_rank'] = config['global_rank'] = 0
    
    if torch.cuda.is_available():
        config['device'] = torch.device("cuda:0")
    else:
        config['device'] = 'cpu'
    
    print(f"✅ Configuration loaded:")
    print(f"  Dataset path: {config['train_data_loader']['dataroot']}")
    print(f"  Device: {config['device']}")
    print(f"  Iterations: {config['trainer']['iterations']}")
    print(f"  Batch size: {config['trainer']['batch_size']}")
    
    try:
        # Initialize trainer (this will load datasets and models)
        print("\n🔧 Initializing trainer...")
        trainer = EnhancedTrainer(config)
        
        print(f"✅ Trainer initialized successfully!")
        print(f"  Train dataset size: {len(trainer.train_dataset)}")
        print(f"  Test dataset size: {len(trainer.test_dataset)}")
        print(f"  Unpaired dataset size: {len(trainer.unpair_dataset)}")
        
        # Test a single forward pass
        print("\n🧪 Testing single forward pass...")
        device = config['device']
        
        # Get a sample from the train loader
        sample_batch = next(iter(trainer.train_loader))
        source_tensor, target_tensor = sample_batch
        source_tensor = source_tensor.to(device)
        target_tensor = target_tensor.to(device)
        
        print(f"  Input shape: {source_tensor.shape}")
        print(f"  Target shape: {target_tensor.shape}")
        
        # Test model forward pass
        trainer.netG.eval()
        with torch.no_grad():
            pred_img, attention = trainer.netG(source_tensor)
            print(f"  Output shape: {pred_img.shape}")
        
        # Test validation
        print("\n🧪 Testing validation...")
        psnr, ssim, lpips = trainer.test(0, lr=0.0002)
        print(f"  Initial validation - PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, LPIPS: {lpips:.4f}")
        
        print("\n✅ All tests passed! Training setup is ready.")
        print("🚀 You can now start training with: python train_subset.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during setup: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_training_setup() 