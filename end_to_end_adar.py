#!/usr/bin/env python3
"""
End-to-End Face Retouching Pipeline
This script performs:
1. Face detection and cropping from input images
2. Face retouching using RetouchFormer model
3. Restoration of retouched faces back to original images
"""

import os
import argparse
import importlib
import shutil
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch
from PIL import Image
from ai_tools.face_tools.face_module.face_module import FACE_DETECTOR_THRESHOLD, FACE_DETECTOR_CHECKPOINT_PATH, \
    FACE_DETECTOR_CHECKPOINT_NAME
from tqdm import tqdm
import cv2

# Import face detection tools from package-ai-tools
from ai_tools.face_tools.face_module import facer


"""
python end_to_end_adar.py --input_dir datasets/data_from_imagen --output_dir datasets/final_test_output --save_face_crops
"""
FACE_DETECTION_CKPT_PTH = "face_detection_checkpoint/mobilenet0.25_Final.pth"


class EndToEndRetouchPipeline:
    def __init__(self, 
                 model_name: str = "RetouchFormer",
                 checkpoint_path: str = "release_model",
                 epoch: str = "best",
                 device: Optional[str] = None,
                 face_size: int = 512,
                 face_detector_threshold: float = 0.5,
                 save_face_crops: bool = False,
                 face_crops_dir: Optional[str] = None,
                 min_change_threshold: int = 4):
        """
        Initialize the end-to-end retouch pipeline
        
        Args:
            model_name: Name of the retouching model
            checkpoint_path: Path to model checkpoint directory
            epoch: Checkpoint epoch to load
            device: Device to run on (cuda:X or cpu)
            face_size: Size to resize face crops to
            face_detector_threshold: Confidence threshold for face detection
            save_face_crops: Whether to save before/after face crop comparisons
            face_crops_dir: Directory to save face crop comparisons (if None, uses output_dir/faces_crop_before_after)
        """
        self.face_size = face_size
        self.face_detector_threshold = face_detector_threshold
        self.save_face_crops = save_face_crops
        self.face_crops_dir = face_crops_dir
        self.min_change_threshold = int(min_change_threshold)
        
        # Setup device
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Initialize face detector
        print("🔄 Initializing face detector...")
        self.face_detector = facer.face_detector(name=FACE_DETECTOR_CHECKPOINT_PATH,
                                                 device=self.device,
                                                 threshold=self.face_detector_threshold,
                                                 model_path=Path(FACE_DETECTION_CKPT_PTH)
                                                 )
        
        # Load RetouchFormer model
        print("🔄 Loading RetouchFormer model...")
        self.model = self._load_retouchformer_model(model_name, checkpoint_path, epoch)
        
    def _load_retouchformer_model(self, model_name: str, checkpoint_path: str, epoch: str):
        """Load the RetouchFormer model"""
        # Set CUDA device BEFORE importing model to avoid custom op conflicts
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
        
        net = importlib.import_module('model.' + model_name)
        model = net.InpaintGenerator().to(self.device)
        
        model_path = f"{checkpoint_path}/gen_{epoch}.pth"
        data = torch.load(model_path, map_location=self.device)
        model.load_state_dict(data)
        print(f'✅ Loaded model from: {model_path}')
        model.eval()
        return model
    
    def detect_and_crop_faces(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict]]:
        """
        Detect faces in image and return cropped faces with their metadata
        
        Args:
            image: Input image as numpy array (H, W, C)
            
        Returns:
            face_crops: List of cropped face images
            face_infos: List of face detection info (bbox, landmarks, etc.)
        """
        # Convert image to tensor format expected by face detector
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        image_tensor = image_tensor.to(self.device)
        
        # Run face detection
        with torch.no_grad():
            faces_data = self.face_detector(image_tensor)
        
        face_crops = []
        face_infos = []
        
        if 'rects' not in faces_data or len(faces_data['rects']) == 0:
            return face_crops, face_infos
        
        # Extract face crops
        for i in range(len(faces_data['rects'])):
            # Get bounding box
            rect = faces_data['rects'][i].cpu().numpy()
            x1, y1, x2, y2 = rect.astype(int)
            
            # Add padding around face
            h, w = image.shape[:2]
            pad = int(max(x2 - x1, y2 - y1) * 0.2)  # 20% padding
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
            
            # Crop face
            face_crop = image[y1:y2, x1:x2]
            face_crops.append(face_crop)
            
            # Store face info for later restoration
            face_info = {
                'bbox': [x1, y1, x2, y2],
                'original_bbox': rect,
                'score': faces_data['scores'][i].cpu().item() if 'scores' in faces_data else 1.0,
                'landmarks': faces_data['points'][i].cpu().numpy() if 'points' in faces_data else None
            }
            face_infos.append(face_info)
        
        return face_crops, face_infos
    
    def preprocess_face_for_model(self, face_image: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Preprocess face crop for RetouchFormer model
        
        Args:
            face_image: Face crop as numpy array (H, W, C)
            
        Returns:
            Tuple of (preprocessed tensor ready for model, resized input face in uint8 RGB)
        """
        # Convert to PIL for resizing
        face_pil = Image.fromarray(face_image)
        face_pil = face_pil.resize((self.face_size, self.face_size), Image.Resampling.LANCZOS)
        
        # Convert back to numpy and normalize
        face_np_float = np.array(face_pil).astype(np.float32) / 255.0
        face_np_uint8 = (face_np_float * 255.0).clip(0, 255).astype(np.uint8)
        
        # Convert to tensor and normalize to [-1, 1]
        face_tensor = torch.from_numpy(face_np_float).permute(2, 0, 1)
        face_tensor = (face_tensor - 0.5) / 0.5
        
        return face_tensor.unsqueeze(0), face_np_uint8
    
    def retouch_face(self, face_tensor: torch.Tensor) -> np.ndarray:
        """
        Apply RetouchFormer model to retouch a face
        
        Args:
            face_tensor: Preprocessed face tensor
            
        Returns:
            Retouched face as numpy array
        """
        face_tensor = face_tensor.to(self.device)
        
        with torch.no_grad():
            retouched_tensor, _ = self.model(face_tensor)
        
        # Convert back to numpy image
        retouched = retouched_tensor[0].cpu()
        retouched = (retouched + 1.0) / 2.0  # [-1, 1] to [0, 1]
        retouched = retouched.permute(1, 2, 0).numpy()
        retouched = (retouched * 255).clip(0, 255).astype(np.uint8)
        
        return retouched
    
    @staticmethod
    def build_change_mask(input_resized_face: np.ndarray, retouched_face: np.ndarray) -> np.ndarray:
        """
        Build a soft change mask highlighting pixels modified by the model.
        
        Args:
            input_resized_face: Original face resized to model input size (uint8 RGB)
            retouched_face: Model output face at model input size (uint8 RGB)
        
        Returns:
            Soft mask in range [0, 1] with shape (H, W, 1)
        """
        # Absolute difference per channel, then convert to grayscale magnitude
        diff = cv2.absdiff(retouched_face, input_resized_face)
        diff_mask = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        diff_mask[diff_mask>1] = 255

        # Smooth the difference map to suppress noise, and use it directly as a soft mask
        diff_mask = cv2.GaussianBlur(diff_mask, (7, 7), 0)
        diff_mask = (diff_mask.astype(np.float32) / 255.0)
        if diff_mask.ndim == 2:
            diff_mask = np.expand_dims(diff_mask, axis=2)
        return diff_mask
    
    def restore_face_to_image(self, original_image: np.ndarray, 
                            retouched_face: np.ndarray,
                            change_mask: np.ndarray,
                            face_info: Dict) -> np.ndarray:
        """
        Restore retouched face back to original image using a data-driven change mask
        
        Args:
            original_image: Original full image
            retouched_face: Retouched face crop (model output, uint8 RGB)
            change_mask: Soft mask (H, W, 1) in [0, 1] indicating changed pixels at model resolution
            face_info: Face detection info containing bbox
            
        Returns:
            Image with retouched face restored
        """
        result_image = original_image.copy()
        
        # Get face region
        x1, y1, x2, y2 = face_info['bbox']
        face_h, face_w = y2 - y1, x2 - x1
        
        # Resize retouched face and mask to original crop size
        retouched_resized = cv2.resize(retouched_face, (face_w, face_h), interpolation=cv2.INTER_LANCZOS4)
        mask_resized = cv2.resize(change_mask, (face_w, face_h), interpolation=cv2.INTER_LINEAR)
        if mask_resized.ndim == 2:
            mask_resized = np.expand_dims(mask_resized, axis=2)
        # Ensure mask has 3 channels for broadcasting
        mask_resized = np.repeat(mask_resized.astype(np.float32), 3, axis=2)

        # Composite only changed pixels back into the original face region
        face_region = result_image[y1:y2, x1:x2]
        blended = (retouched_resized.astype(np.float32) * mask_resized +
                   face_region.astype(np.float32) * (1.0 - mask_resized)).astype(np.uint8)
        result_image[y1:y2, x1:x2] = blended
        
        return result_image
    
    def create_face_comparison(self, original_face: np.ndarray, retouched_face: np.ndarray,
                               change_mask: np.ndarray, output_path: str, face_idx: int):
        """
        Create a side-by-side comparison of original, retouched, and mask with labels.
        
        Args:
            original_face: Original face crop
            retouched_face: Retouched face crop
            change_mask: Soft mask (H, W, 1) in [0, 1] indicating changed pixels at model resolution
            output_path: Path to save comparison image
            face_idx: Face index for filename
        """
        try:
            # Ensure both faces are the same size
            h, w = original_face.shape[:2]
            retouched_resized = cv2.resize(retouched_face, (w, h), interpolation=cv2.INTER_LANCZOS4)
            # Prepare mask visualization (resize to face size, apply colormap)
            mask_2d = change_mask.squeeze()
            if mask_2d.ndim == 0:
                mask_2d = np.zeros((self.face_size, self.face_size), dtype=np.float32)
            mask_resized_2d = cv2.resize(mask_2d, (w, h), interpolation=cv2.INTER_LINEAR)
            mask_gray_vis = (np.clip(mask_resized_2d, 0.0, 1.0) * 255.0).astype(np.uint8)
            mask_color_vis = cv2.applyColorMap(mask_gray_vis, cv2.COLORMAP_JET)
            
            # Create side-by-side comparison
            comparison_width = w * 3
            comparison_height = h + 40  # Extra space for text labels
            comparison = np.ones((comparison_height, comparison_width, 3), dtype=np.uint8) * 255
            
            # Convert RGB to BGR for saving (OpenCV uses BGR)
            original_face_bgr = cv2.cvtColor(original_face, cv2.COLOR_RGB2BGR)
            retouched_resized_bgr = cv2.cvtColor(retouched_resized, cv2.COLOR_RGB2BGR)
            
            # Place images
            comparison[40:40+h, 0:w] = original_face_bgr
            comparison[40:40+h, w:w*2] = retouched_resized_bgr
            comparison[40:40+h, w*2:w*3] = mask_color_vis
            
            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_thickness = 2
            text_color = (0, 0, 0)  # Black text
            
            # Calculate text positions (centered)
            before_text = "BEFORE"
            after_text = "AFTER"
            mask_text = "MASK"
            
            # Get text size for centering
            (before_w, before_h), _ = cv2.getTextSize(before_text, font, font_scale, font_thickness)
            (after_w, after_h), _ = cv2.getTextSize(after_text, font, font_scale, font_thickness)
            (mask_w, mask_h), _ = cv2.getTextSize(mask_text, font, font_scale, font_thickness)
            
            # Position text
            before_x = (w - before_w) // 2
            after_x = w + (w - after_w) // 2
            mask_x = w * 2 + (w - mask_w) // 2
            text_y = 25
            
            cv2.putText(comparison, before_text, (before_x, text_y), font, font_scale, text_color, font_thickness)
            cv2.putText(comparison, after_text, (after_x, text_y), font, font_scale, text_color, font_thickness)
            cv2.putText(comparison, mask_text, (mask_x, text_y), font, font_scale, text_color, font_thickness)
            
            # Save comparison
            filename = f"face_{face_idx + 1}.jpg"
            cv2.imwrite(str(Path(output_path) / filename), comparison)
            # Save raw mask (grayscale) alongside
            mask_filename = f"face_{face_idx + 1}_mask.png"
            cv2.imwrite(str(Path(output_path) / mask_filename), mask_gray_vis)
            
        except Exception as e:
            print(f"⚠️  Failed to create face comparison: {e}")
    
    def process_image(self, image_path: str, output_path: str, face_crops_output_base: Optional[str] = None) -> Tuple[bool, int]:
        """
        Process a single image: detect faces, retouch them, and save result
        
        Args:
            image_path: Path to input image
            output_path: Path to save output image
            face_crops_output_base: Base directory for face crop comparisons
            
        Returns:
            Tuple of (success_status, number_of_faces_processed)
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Failed to load image: {image_path}")
                return False, 0
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect and crop faces
            face_crops, face_infos = self.detect_and_crop_faces(image_rgb)
            
            if len(face_crops) == 0:
                print(f"⚠️  No faces detected in: {image_path}")
                # Save original image
                cv2.imwrite(output_path, image)
                return True, 0

            # Setup face crops directory if needed
            face_crops_dir = None
            if self.save_face_crops and face_crops_output_base:
                image_name = Path(image_path).stem  # filename without extension
                face_crops_dir = Path(face_crops_output_base) / image_name
                face_crops_dir.mkdir(parents=True, exist_ok=True)
            
            # Process each face
            result_image = image_rgb.copy()
            retouched_faces = []
            
            for i, (face_crop, face_info) in enumerate(zip(face_crops, face_infos)):
                # Preprocess face (also obtain resized input for mask computation)
                face_tensor, resized_input_face = self.preprocess_face_for_model(face_crop)
                
                # Retouch face
                retouched_face = self.retouch_face(face_tensor)
                retouched_faces.append(retouched_face)
                
                # Build change mask at model resolution
                change_mask = self.build_change_mask(resized_input_face, retouched_face)
                
                # Save face comparison if enabled (include mask and save raw mask)
                if self.save_face_crops and face_crops_dir:
                    self.create_face_comparison(face_crop, retouched_face, change_mask, str(face_crops_dir), i)

                # If mask is effectively empty (all zeros), skip restoration
                if not np.any(change_mask > 0):
                    continue

                # Restore to original image using change mask
                result_image = self.restore_face_to_image(result_image, retouched_face, change_mask, face_info)
            
            # Convert back to BGR and save
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, result_bgr)
            
            faces_msg = f"Processed {len(face_crops)} face(s) in: {Path(image_path).name}"
            if self.save_face_crops and face_crops_dir:
                faces_msg += f" (comparisons saved to {face_crops_dir.name}/)"
            print(faces_msg)
            return True, len(face_crops)
            
        except Exception as e:
            print(f"❌ Error processing {image_path}: {str(e)}")
            return False, 0
    
    def process_directory(self, input_dir: str, output_dir: str, 
                         image_extensions: List[str] = None):
        """
        Process all images in a directory
        
        Args:
            input_dir: Input directory path
            output_dir: Output directory path
            image_extensions: List of image file extensions to process
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Setup face crops directory if needed
        face_crops_output_base = None
        if self.save_face_crops:
            if self.face_crops_dir:
                face_crops_output_base = self.face_crops_dir
            else:
                face_crops_output_base = str(output_path / "faces_crop_before_after")
            Path(face_crops_output_base).mkdir(parents=True, exist_ok=True)
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f'*{ext}'))
            image_files.extend(input_path.glob(f'*{ext.upper()}'))
        
        if len(image_files) == 0:
            print(f"No image files found in: {input_dir}")
            return
        
        print(f"Found {len(image_files)} images to process")
        if self.save_face_crops:
            print(f"Face crop comparisons will be saved to: {face_crops_output_base}")
        
        # Start timing
        start_time = time.time()
        
        # Process each image
        success_count = 0
        total_faces = 0
        for image_file in tqdm(image_files, desc="Processing images"):
            # Copy original image to output directory
            original_output_file = output_path / image_file.name
            shutil.copy2(str(image_file), str(original_output_file))
            
            # Create output filename with _output suffix
            stem = image_file.stem  # filename without extension
            suffix = image_file.suffix  # file extension
            output_filename = f"{stem}_output{suffix}"
            output_file = output_path / output_filename
            
            success, faces_count = self.process_image(str(image_file), str(output_file), face_crops_output_base)
            if success:
                success_count += 1
                total_faces += faces_count
        
        # Calculate timing
        end_time = time.time()
        total_time = end_time - start_time
        
        print(f"\nPsrocessing complete! Successfully processed {success_count}/{len(image_files)} images")
        print(f"Total faces processed: {total_faces}")
        print(f"Total time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        if success_count > 0:
            print(f"Average time per image: {total_time/success_count:.2f} seconds")
        if total_faces > 0:
            print(f"Average time per face: {total_time/total_faces:.2f} seconds")
        print(f"Original images copied to: {output_dir}")
        print(f"Retouched images saved as: *_output{image_files[0].suffix} in {output_dir}")
        if self.save_face_crops:
            print(f"Face comparisons saved to: {face_crops_output_base}")


def main():
    parser = argparse.ArgumentParser(description="End-to-End Face Retouching Pipeline")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Path to input images directory")
    parser.add_argument("--output_dir", type=str,
                       help="Path to output images directory")
    parser.add_argument("--checkpoint_path", type=str, default="release_model",
                       help="Path to model checkpoint directory")
    parser.add_argument("--epoch", type=str, default="best",
                       help="Checkpoint epoch to load, also support str like 'best' and 'last'")
    parser.add_argument("--device", type=str, default=None,
                       help="Device to use (e.g., cuda:0, cuda:1, cpu)")
    parser.add_argument("--extensions", nargs='+', default=None,
                       help="Image file extensions to process")
    parser.add_argument("--save_face_crops", action="store_true",
                       help="Save before/after face crop comparisons")
    parser.add_argument("--face_crops_dir", type=str, default=None,
                       help="Custom directory for face crop comparisons (default: output_dir/faces_crop_before_after)")
    
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"{args.input_dir}_output"
        os.makedirs(args.output_dir, exist_ok=True)
    
    print("RetouchFormer End-to-End Face Retouching Pipeline")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    if args.save_face_crops:
        crops_dir = args.face_crops_dir or f"{args.output_dir}/faces_crop_before_after"
        print(f"Face crops will be saved to: {crops_dir}")

    # Initialize pipeline
    pipeline = EndToEndRetouchPipeline(
        checkpoint_path=args.checkpoint_path,
        epoch=args.epoch,
        device=args.device,
        save_face_crops=args.save_face_crops,
        face_crops_dir=args.face_crops_dir
    )
    
    # Process directory
    pipeline.process_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        image_extensions=args.extensions
    )


if __name__ == "__main__":
    main()