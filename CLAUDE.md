# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

RetouchFormer is a semi-supervised face retouching transformer that removes imperfections from face images while preserving content. It was presented at AAAI 2024 and uses prior-based selective self-attention for high-quality face retouching.

## Common Development Commands

### Environment Setup
```bash
# Create and activate conda environment
conda create -n retouchformer python=3.8
conda activate retouchformer
pip install -r requirements.txt
```

### Training Commands
```bash
# Train with default configuration
python train.py

# Train with specific configuration
python train.py -c ./configs/RetouchFormer_trainer.json

# Train on subset (faster iteration)
python train_subset.py

# Train full model
python train_full.py

# Train enhanced full model
python train_enhanced_full.py
```

### Inference Commands
```bash
# Run inference on test images
python img_retouching.py --input_path datasets/test --save_path results

# Run inference with specific checkpoint
python img_retouching.py -e best -c release_model --input_path datasets/test

# Run inference with metrics evaluation
python inference_with_metrics.py --output_dir results_with_metrics

# Run end-to-end face retouching (automatic face detection + retouching)
python end_to_end_adar.py --input_dir /path/to/images --output_dir /path/to/output

# With face crop comparisons saved
python end_to_end_adar.py --input_dir /path/to/images --output_dir /path/to/output --save_face_crops
```

### Evaluation Commands
```bash
# Evaluate model performance
python eval.py -e best -c release_model --model RetouchFormer

# Evaluate with authors' model
python adar_evaluation/evaluate_authors_model.py

# Compare model results
python adar_evaluation/compare_results.py
```

### Monitoring Training
```bash
# Launch tensorboard for training visualization
tensorboard --logdir checkpoints_subset/RetouchFormer_RetouchFormer_subset_trainer/tensorboard
```

### Testing Setup
```bash
# Verify training setup before starting
python test_training_setup.py
```

## Architecture Overview

### Core Components

1. **Model Architecture** (`model/`)
   - `RetouchFormer.py`: Main model implementation with InpaintGenerator and discriminator
   - `network_vrt_pair_qkv.py`: Vision Transformer stages with QKV attention mechanisms
   - `gpen_model.py`: Encoder/Decoder components for feature extraction
   - Combines transformer-based attention with GAN framework for face retouching

2. **Training Pipeline** (`core/`)
   - `trainer.py`: Base trainer class handling the training loop
   - `trainer_enhanced.py`: Enhanced trainer with improved tensorboard logging
   - `dataset.py`: FaceRetouchingDataset for loading paired source/target images
   - `loss.py`: Multiple loss functions including L1, VGG perceptual, LPIPS, ID, and adversarial losses

3. **Loss Functions**
   - Valid loss (reconstruction)
   - Mask loss (imperfection localization)
   - VGG perceptual loss
   - LPIPS perceptual loss
   - ID loss (face identity preservation)
   - Adversarial loss (GAN)
   - SSIM loss (structural similarity)

4. **Dataset Structure**
   ```
   face_retouching/
   ├── train/
   │   ├── source/  # Imperfect face images
   │   └── target/  # Retouched face images
   └── test/
       ├── source/
       └── target/
   ```

### Key Configuration Parameters

Located in `configs/*.json`:
- `iterations`: Total training iterations (default: 500k)
- `batch_size`: Training batch size
- `lr`: Learning rate (default: 2e-4)
- `size`: Input image size (default: 512)
- Loss weights for balancing multiple objectives

### Model Workflow

1. **Imperfection Detection**: Uses reconstruction-oriented localization to identify blemishes
2. **Selective Attention**: Applies attention between imperfection queries and normal skin key-values
3. **Content Synthesis**: Generates clean skin content to replace imperfections
4. **Multi-scale Processing**: Handles imperfections at various scales

## Important Notes

- Pre-trained model checkpoint `gen_best.pth` should be placed in `release_model/`
- GPU is required (CUDA visible devices are set in scripts)
- Default image resolution is 512x512
- The model uses distributed training support but defaults to single GPU
- Validation frequency and checkpoint saving intervals are configurable

## End-to-End Face Retouching Pipeline

The `end_to_end_adar.py` script provides automatic face detection and retouching for any images:

### Features
- Automatic face detection using RetinaFace (from package-ai-tools)
- Handles multiple faces per image
- Seamless blending of retouched faces back to original images
- Supports various image formats (.jpg, .png, .tif, etc.)
- Optional face crop before/after comparisons with text labels

### Usage
```bash
# Basic usage
python end_to_end_adar.py --input_dir /path/to/images --output_dir /path/to/output

# With custom options and face crop comparisons
python end_to_end_adar.py \
    --input_dir /path/to/images \
    --output_dir /path/to/output \
    --device cuda:0 \
    --face_size 512 \
    --face_threshold 0.5 \
    --extensions .jpg .png \
    --save_face_crops \
    --face_crops_dir /custom/path/for/crops
```

### Pipeline Workflow
1. Detects all faces in input images using RetinaFace
2. Crops each face with 20% padding for context
3. Resizes faces to 512x512 and preprocesses for RetouchFormer
4. Applies face retouching model
5. **Optional**: Saves before/after face crop comparisons with text labels
6. Blends retouched faces back using elliptical masks with Gaussian blur
7. Saves both original and retouched images to output directory
   - Original images: `image_name.ext` (exact copy)
   - Retouched images: `image_name_output.ext`

### Face Crop Comparisons Structure
When `--save_face_crops` is enabled, the following structure is created:
```
output_dir/
├── image1.jpg                      # Original image (copied)
├── image1_output.jpg              # Retouched image
├── image2.jpg                      # Original image (copied)
├── image2_output.jpg              # Retouched image
└── faces_crop_before_after/        # Optional (with --save_face_crops)
    ├── image1/                     # Directory per image (no extension)
    │   ├── face_1.jpg             # Side-by-side before/after comparison
    │   └── face_2.jpg             # Additional faces if present
    └── image2/
        └── face_1.jpg
```

### Requirements
- `package-ai-tools` must be installed: `pip install package-ai-tools`
- Pre-trained face detection model downloads automatically on first run

## Current Branch Status

Working on branch `adar-research` with several uncommitted changes including:
- New training configurations for full and enhanced models
- Additional evaluation scripts and results
- Enhanced trainer implementation with better logging
- End-to-end face retouching pipeline (`end_to_end_adar.py`)