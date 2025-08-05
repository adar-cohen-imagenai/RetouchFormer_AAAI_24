# Adar Evaluation Suite

This directory contains all evaluation scripts and results for comparing RetouchFormer models.

## 📁 Directory Structure

```
adar_evaluation/
├── Scripts/
│   ├── evaluate_model.py           # Evaluate enhanced models (with config)
│   ├── evaluate_authors_model.py   # Evaluate authors' pre-trained model
│   ├── create_fat_comparison.py    # Generate side-by-side comparison images
│   ├── compare_results.py          # Compare evaluation metrics
│   └── compare_models.py           # Multi-model comparison tool
├── Results/
│   ├── evaluation_authors/         # Authors' model evaluation results
│   ├── evaluation_enhanced_70k/    # Enhanced model evaluation results
│   └── fat_comparisons/            # Fat comparison images
└── README.md                       # This file
```

## 🧪 Evaluation Results Summary

### Model Comparison (50 test samples)
| Metric | Authors' Model | Enhanced Model (70k) | Difference |
|--------|----------------|---------------------|------------|
| **PSNR** | 42.81 ± 3.30 dB | 42.35 ± 3.14 dB | -0.46 dB |
| **SSIM** | 0.993 ± 0.004 | 0.992 ± 0.005 | -0.001 |
| **LPIPS** | 0.010 ± 0.009 | 0.013 ± 0.010 | +0.003 |

**✅ Assessment**: Enhanced model performs **COMPETITIVELY** with authors' model.

## 🖼️ Image Outputs

### Fat Comparison Images (`fat_comparisons/`)
- **Format**: `Input | Target | Authors' Output | Enhanced Output`
- **Titles**: Each section has a clear title showing the model/checkpoint name
- **Count**: 20 comparison images
- **Naming**: `sample_XXXXX_fat_comparison.jpg`

### Individual Model Results
- **Authors' model**: `evaluation_authors/` (50 images)
- **Enhanced model**: `evaluation_enhanced_70k/` (50 images)
- **Format**: Both comparison and individual prediction images

## 🛠️ Script Usage

### 1. Evaluate Authors' Model
```bash
python adar_evaluation/evaluate_authors_model.py \
    --checkpoint_path "release_model/gen_best.pth" \
    --device cuda:0 \
    --save_images \
    --max_samples 50
```

### 2. Evaluate Enhanced Model
```bash
python adar_evaluation/evaluate_model.py \
    --checkpoint_dir "checkpoints_enhanced_full/RetouchFormer_RetouchFormer_enhanced_full_trainer" \
    --checkpoint_file "gen_070000.pth" \
    --device cuda:0 \
    --save_images \
    --max_samples 50
```

### 3. Generate Fat Comparison Images
```bash
python adar_evaluation/create_fat_comparison.py \
    --authors_checkpoint "release_model/gen_best.pth" \
    --enhanced_checkpoint_dir "checkpoints_enhanced_full/RetouchFormer_RetouchFormer_enhanced_full_trainer" \
    --enhanced_checkpoint_file "gen_070000.pth" \
    --device cuda:0 \
    --max_samples 20
```

### 4. Compare Results
```bash
python adar_evaluation/compare_results.py \
    "adar_evaluation/evaluation_authors/evaluation_metrics.json" \
    "adar_evaluation/evaluation_enhanced_70k/evaluation_metrics.json" \
    --model_names "Authors' Model" "Enhanced Model (70k)"
```

## 📊 Key Features

### ✅ All scripts support `--max_samples` parameter
- **Purpose**: Limit evaluation samples for faster testing
- **Usage**: `--max_samples 100` (or any number)
- **Default**: `None` (processes all samples)

### ✅ Organized output structure
- **All results**: Saved within `adar_evaluation/` directory
- **No external clutter**: Old evaluation directories cleaned up
- **Clear naming**: Model names included in titles and file paths

### ✅ Fat comparison images
- **Layout**: Input | Target | Authors' Output | Enhanced Output
- **Titles**: Clear labels showing model/checkpoint information
- **Quality**: High-quality JPEG images with proper color handling

## 🎯 Results Interpretation

### Performance Assessment
- **Both models achieve excellent quality** (PSNR > 42 dB, SSIM > 0.99)
- **Differences are minimal** and within expected ranges
- **Enhanced model performs competitively** with published results

### Visual Quality
- Check the **fat comparison images** for visual assessment
- Compare texture detail, color accuracy, and artifact presence
- Both models show high-quality face retouching results

## 📁 File Types

| Extension | Description |
|-----------|-------------|
| `.json` | Evaluation metrics and metadata |
| `_comparison.jpg` | Side-by-side: Source \| Prediction \| Target |
| `_prediction.jpg` | Model prediction only |
| `_fat_comparison.jpg` | Four-way: Input \| Target \| Authors \| Enhanced |

---

**Note**: All scripts are designed to work from the project root directory and automatically handle import paths. 