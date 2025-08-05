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
import cv2
from PIL import Image, ImageDraw, ImageFont

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

def load_enhanced_model(checkpoint_path, config_path, device):
    """Load enhanced model from checkpoint"""
    print(f"🔄 Loading enhanced model from: {checkpoint_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
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
    
    print(f"✅ Enhanced model loaded successfully")
    return model

def add_title_to_image(image, title, font_size=24):
    """Add title text to the top of an image"""
    # Convert to PIL if it's numpy
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    else:
        pil_image = image
    
    # Create a new image with extra space for title
    title_height = font_size + 20  # padding
    new_width = pil_image.width
    new_height = pil_image.height + title_height
    
    # Create new image with white background for title area
    new_image = Image.new('RGB', (new_width, new_height), color='white')
    new_image.paste(pil_image, (0, title_height))
    
    # Add title text
    draw = ImageDraw.Draw(new_image)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        font = ImageFont.load_default()
    
    # Center the text
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_x = (new_width - text_width) // 2
    text_y = 10
    
    draw.text((text_x, text_y), title, fill='black', font=font)
    
    return new_image

def create_fat_comparison(source_np, target_np, pred_authors_np, pred_enhanced_np, 
                         authors_name, enhanced_name, save_path):
    """Create fat comparison image with titles"""
    
    # Normalize images to [0, 1] range
    images = []
    titles = ["Input", "Target", authors_name, enhanced_name]
    
    for img_np in [source_np, target_np, pred_authors_np, pred_enhanced_np]:
        # Ensure [0, 1] range
        img_normalized = np.clip(img_np, 0, 1)
        # Convert to [0, 255] and transpose to HWC
        img_uint8 = (img_normalized.transpose(1, 2, 0) * 255).astype(np.uint8)
        # Image is already in RGB format from PyTorch, no need for BGR conversion
        images.append(img_uint8)
    
    # Add titles to each image
    titled_images = []
    for img, title in zip(images, titles):
        titled_img = add_title_to_image(img, title, font_size=20)
        titled_images.append(titled_img)
    
    # Concatenate horizontally
    total_width = sum(img.width for img in titled_images)
    max_height = max(img.height for img in titled_images)
    
    fat_image = Image.new('RGB', (total_width, max_height), color='white')
    
    x_offset = 0
    for img in titled_images:
        fat_image.paste(img, (x_offset, 0))
        x_offset += img.width
    
    # Save the fat comparison image
    fat_image.save(save_path, quality=95)

def generate_fat_comparisons(authors_model, enhanced_model, test_loader, device, 
                           output_dir, authors_name, enhanced_name, max_samples=None):
    """Generate fat comparison images for test samples"""
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"💾 Saving fat comparison images to: {output_dir}")
    
    processed_samples = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Creating fat comparisons")):
            # Check if we've reached max_samples limit
            if max_samples is not None and processed_samples >= max_samples:
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
            
            # Forward pass through both models
            pred_authors_tensor, _ = authors_model(source_tensor)
            pred_enhanced_tensor, _ = enhanced_model(source_tensor)
            
            # Convert to numpy for image processing
            source_np = source_tensor[0].cpu().numpy()
            target_np = target_tensor[0].cpu().numpy()
            pred_authors_np = pred_authors_tensor[0].cpu().numpy()
            pred_enhanced_np = pred_enhanced_tensor[0].cpu().numpy()
            
            # Normalize from [-1, 1] to [0, 1]
            source_np = (source_np + 1.0) / 2.0
            target_np = (target_np + 1.0) / 2.0
            pred_authors_np = (pred_authors_np + 1.0) / 2.0
            pred_enhanced_np = (pred_enhanced_np + 1.0) / 2.0
            
            # Clip values to valid range
            source_np = np.clip(source_np, 0, 1)
            target_np = np.clip(target_np, 0, 1)
            pred_authors_np = np.clip(pred_authors_np, 0, 1)
            pred_enhanced_np = np.clip(pred_enhanced_np, 0, 1)
            
            # Create fat comparison image
            save_path = os.path.join(output_dir, f"{name}_fat_comparison.jpg")
            create_fat_comparison(
                source_np, target_np, pred_authors_np, pred_enhanced_np,
                authors_name, enhanced_name, save_path
            )
            
            processed_samples += 1
    
    print(f"✅ Generated {processed_samples} fat comparison images")

def main():
    parser = argparse.ArgumentParser(description='Create Fat Comparison Images with Both Models')
    parser.add_argument('--authors_checkpoint', type=str, default='release_model/gen_best.pth',
                       help='Path to authors\' checkpoint file')
    parser.add_argument('--enhanced_checkpoint_dir', type=str, 
                       default='checkpoints_enhanced_full/RetouchFormer_RetouchFormer_enhanced_full_trainer',
                       help='Directory containing enhanced model checkpoint and config')
    parser.add_argument('--enhanced_checkpoint_file', type=str, default='gen_070000.pth',
                       help='Enhanced model checkpoint filename')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to run evaluation on')
    parser.add_argument('--output_dir', type=str, default='adar_evaluation/fat_comparisons',
                       help='Directory to save fat comparison images')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (None for all)')
    parser.add_argument('--dataset_path', type=str, 
                       default='/data/data/face_retouch/face_retouching_ffhqr_dataset',
                       help='Path to test dataset')
    
    args = parser.parse_args()
    
    print("🎯 Fat Comparison Image Generator")
    print("="*60)
    
    # Setup device
    if torch.cuda.is_available() and 'cuda' in args.device:
        device = torch.device(args.device)
        print(f"🚀 Using device: {device}")
    else:
        device = torch.device('cpu')
        print("💻 Using CPU")
    
    # Check if checkpoints exist
    if not os.path.exists(args.authors_checkpoint):
        raise FileNotFoundError(f"Authors' checkpoint not found: {args.authors_checkpoint}")
    
    enhanced_checkpoint_path = os.path.join(args.enhanced_checkpoint_dir, args.enhanced_checkpoint_file)
    if not os.path.exists(enhanced_checkpoint_path):
        raise FileNotFoundError(f"Enhanced checkpoint not found: {enhanced_checkpoint_path}")
    
    # Find config file for enhanced model
    config_files = [f for f in os.listdir(args.enhanced_checkpoint_dir) if f.endswith('.json')]
    if not config_files:
        raise FileNotFoundError(f"No config file found in {args.enhanced_checkpoint_dir}")
    config_path = os.path.join(args.enhanced_checkpoint_dir, config_files[0])
    
    # Load models
    authors_model = load_authors_model(args.authors_checkpoint, device)
    enhanced_model = load_enhanced_model(enhanced_checkpoint_path, config_path, device)
    
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
    
    # Display sample info
    if args.max_samples is not None:
        print(f"📊 Test dataset loaded: {len(test_dataset)} samples (processing {args.max_samples})")
    else:
        print(f"📊 Test dataset loaded: {len(test_dataset)} samples")
    
    # Generate model names for titles
    authors_name = f"Authors ({os.path.basename(args.authors_checkpoint)})"
    enhanced_name = f"Enhanced ({args.enhanced_checkpoint_file})"
    
    print(f"\n🏷️  Model names for titles:")
    print(f"   Authors: {authors_name}")
    print(f"   Enhanced: {enhanced_name}")
    
    # Generate fat comparison images
    generate_fat_comparisons(
        authors_model=authors_model,
        enhanced_model=enhanced_model,
        test_loader=test_loader,
        device=device,
        output_dir=args.output_dir,
        authors_name=authors_name,
        enhanced_name=enhanced_name,
        max_samples=args.max_samples
    )
    
    print(f"\n💾 Fat comparison images saved to: {args.output_dir}")
    print("🎨 Each image shows: Input | Target | Authors' Output | Enhanced Output")

if __name__ == "__main__":
    main() 