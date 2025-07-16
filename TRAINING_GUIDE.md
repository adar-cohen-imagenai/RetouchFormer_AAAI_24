# 🎯 RetouchFormer Training Guide

This guide explains how to train RetouchFormer on your subset dataset with enhanced tensorboard logging.

## 📁 What We've Created

### Files Created:
- `configs/RetouchFormer_subset_trainer.json` - Training configuration for subset dataset
- `core/trainer_enhanced.py` - Enhanced trainer with better tensorboard logging
- `train_subset.py` - Training script for subset dataset
- `test_training_setup.py` - Script to test setup before training
- `inference_with_metrics.py` - Single script for inference + metrics + comparison images

### Key Features:
- ✅ **Separate Train/Val Tensorboard Writers** - View training and validation metrics on same graphs
- ✅ **Frequent Validation** - Every 50 iterations (configurable)
- ✅ **Short Training Loop** - 2000 iterations suitable for subset data
- ✅ **Enhanced Logging** - Better progress tracking and visualization
- ✅ **Comprehensive Metrics** - PSNR, SSIM, LPIPS evaluation

## 🚀 Quick Start

### 1. Test Setup
First, verify everything works:
```bash
python test_training_setup.py
```
**Expected Output:**
```
✅ All tests passed! Training setup is ready.
🚀 You can now start training with: python train_subset.py
```

### 2. Start Training
```bash
python train_subset.py
```

### 3. Monitor Training
Open tensorboard in another terminal:
```bash
tensorboard --logdir checkpoints_subset/RetouchFormer_RetouchFormer_subset_trainer/tensorboard
```
Then open: http://localhost:6006

## 📊 Tensorboard Monitoring

You'll see **two separate tabs**:
- **train/** - Training losses (L1, VGG, GAN, etc.)
- **val/** - Validation metrics (PSNR, SSIM, LPIPS)

### Key Metrics to Watch:
- **train/loss/l1_loss** - Main reconstruction loss (should decrease)
- **val/PSNR** - Peak Signal-to-Noise Ratio (should increase, >20 is good)
- **val/SSIM** - Structural Similarity (should increase, >0.8 is good) 
- **val/LPIPS** - Perceptual quality (should decrease, <0.3 is good)

## 🎛️ Configuration Options

Edit `configs/RetouchFormer_subset_trainer.json`:

```json
{
    "trainer": {
        "iterations": 2000,     // Total training iterations
        "log_freq": 10,         // How often to log training metrics
        "val_freq": 50,         // How often to run validation
        "save_freq": 100,       // How often to save checkpoints
        "lr": 2e-4,            // Learning rate
        "batch_size": 1         // Batch size
    }
}
```

## 📈 Expected Training Progress

### Initial (Iteration 0):
- PSNR: ~8-10 dB (very poor)
- SSIM: ~0.1-0.2 (very poor)
- LPIPS: ~0.9 (very poor)

### Good Training (After 1000+ iterations):
- PSNR: >25 dB (good improvement)
- SSIM: >0.8 (good structure)
- LPIPS: <0.5 (decent perceptual quality)

### Excellent Training (After 2000 iterations):
- PSNR: >30 dB (excellent)
- SSIM: >0.9 (excellent structure)
- LPIPS: <0.3 (good perceptual quality)

## 🔄 Inference After Training

After training, use your new checkpoint:
```bash
# Test with your trained model
python inference_with_metrics.py --output_dir results_from_my_training

# Compare with pretrained model
python inference_with_metrics.py --model_path release_model/gen_best.pth --output_dir results_pretrained
```

## 🎯 Comparing Performance

The goal is to see if your trained model performs **similarly** to the pretrained model:

1. **Visual Quality**: Check the comparison images side-by-side
2. **Quantitative Metrics**: Compare PSNR/SSIM/LPIPS values
3. **Training Curves**: Smooth learning curves indicate good training

## 📂 Output Structure

After training you'll have:
```
checkpoints_subset/
├── RetouchFormer_RetouchFormer_subset_trainer/
│   ├── tensorboard/
│   │   ├── train/          # Training metrics
│   │   └── val/            # Validation metrics
│   ├── gen_000100.pth      # Generator checkpoints
│   ├── dis_000100.pth      # Discriminator checkpoints
│   └── opt_000100.pth      # Optimizer states
└── RetouchFormer_subset.txt # Training log
```

## 🛠️ Troubleshooting

### Common Issues:

1. **CUDA Out of Memory**: Reduce batch_size in config
2. **Training Stalls**: Check if learning rate is too high/low
3. **No Improvement**: Ensure dataset paths are correct
4. **Tensorboard Not Updating**: Refresh browser or restart tensorboard

### Debug Commands:
```bash
# Check dataset loading
python -c "from core.dataset import FaceRetouchingDataset; d=FaceRetouchingDataset('/data/data/face_retouch/face_retouching_subset'); print(len(d))"

# Check GPU usage
nvidia-smi

# Monitor training files
ls -la checkpoints_subset/RetouchFormer_RetouchFormer_subset_trainer/
```

## 🎉 Success Criteria

Your training is successful if:
- ✅ Loss curves are smooth and decreasing
- ✅ Validation PSNR > 25 dB after 1000 iterations
- ✅ Visual quality of outputs looks reasonable
- ✅ Model checkpoints are being saved regularly

Good luck with your training! 🚀 