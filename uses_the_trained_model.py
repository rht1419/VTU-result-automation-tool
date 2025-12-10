#!/usr/bin/env python3
"""
CAPTCHA Recognition Module using trained PyTorch model
Replaces the Gemini CAPTCHA service with local model inference
"""

import torch
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
from typing import Optional

def get_project_paths():
    """
    Get the correct project paths regardless of where the script is running from.
    """
    # Get the current file's directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up to find the project root (should be the parent of vtu automate 2)
    project_root = os.path.dirname(current_dir)
    
    # Verify we have the right directory by checking for key files
    if not os.path.exists(os.path.join(project_root, "data_19k")):
        # Try going up one more level
        project_root = os.path.dirname(project_root)
    
    # If still not found, use current directory
    if not os.path.exists(os.path.join(project_root, "data_19k")):
        project_root = os.path.dirname(os.path.abspath(__file__))
    
    model_path = os.path.join(project_root, "runs", "exp_19k", "best_epoch_134_fullacc_0.9951.pth")
    train_labels_path = os.path.join(project_root, "data_19k", "train_labels.txt")
    
    return project_root, model_path, train_labels_path

# Get paths
project_root, model_path, train_labels_path = get_project_paths()

# Add project root to sys.path to import train_captcha_19k
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    # Import required components from the training script
    from train_captcha_19k import (
        MultiHeadResNet, get_transforms, load_label_list, get_valid_charset
    )
except ImportError as e:
    print(f"Error importing training modules: {e}")
    print("Make sure you're running this from the correct directory")
    print(f"Project root: {project_root}")
    print(f"Model path: {model_path}")
    print(f"Train labels path: {train_labels_path}")
    raise

class CaptchaModelRecognizer:
    def __init__(self, model_path=model_path):
        """
        Initialize the CAPTCHA recognizer with the trained model
        
        Args:
            model_path (str): Path to the trained model checkpoint
        """
        self.model_path = model_path
        self.train_labels_path = train_labels_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load character mappings from training data
        train_list = load_label_list(self.train_labels_path)
        self.valid_chars = get_valid_charset(train_list)
        self.char2idx = {c: i for i, c in enumerate(self.valid_chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.num_classes = len(self.valid_chars)
        
        print(f"✅ Loaded character set: {len(self.valid_chars)} characters")
        print(f"🔤 Characters: {''.join(self.valid_chars)}")
        
        # Initialize model
        self.model = MultiHeadResNet(num_heads=6, num_classes=self.num_classes, pretrained=False).to(self.device)
        
        # Load trained weights
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found at {self.model_path}")
            
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()
        
        epoch_info = checkpoint.get('epoch', 'unknown')
        accuracy_info = checkpoint.get('best_val', 0.0)
        print(f"✅ Loaded model from epoch {epoch_info} with accuracy {accuracy_info:.4f}")
        
        # Get validation transform (no augmentation)
        self.val_transform = get_transforms("val")
        
    def preprocess_image(self, image_path):
        """
        Preprocess the CAPTCHA image for model inference
        
        Args:
            image_path (str): Path to the CAPTCHA image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load and preprocess image
        img = Image.open(image_path).convert("RGB")  # Load as RGB first
        img_gray = img.convert("L")  # Convert to grayscale
        
        # Resize to training size (128, 64)
        img_resized = img_gray.resize((128, 64))
        arr = np.array(img_resized).astype(np.float32) / 255.0
        
        # Convert to uint8 for albumentations
        arr8 = (arr * 255).astype(np.uint8)
        
        # Apply validation transform
        augmented = self.val_transform(image=arr8)
        img_tensor = augmented["image"].unsqueeze(0).to(self.device)  # Add batch dimension
        
        return img_tensor
    
    def predict(self, image_path):
        """
        Predict the CAPTCHA text from an image
        
        Args:
            image_path (str): Path to the CAPTCHA image
            
        Returns:
            str: Predicted CAPTCHA text (6 characters)
        """
        try:
            # Preprocess image
            img_tensor = self.preprocess_image(image_path)
            
            # Predict with model
            with torch.no_grad():
                logits = self.model(img_tensor)  # [1, num_heads, num_classes]
                probs = torch.softmax(logits, dim=-1)
                preds = logits.argmax(dim=-1).squeeze(0)  # [num_heads]
                
            # Convert predictions to characters
            predicted_chars = []
            for i in range(len(preds)):
                char = self.idx2char[preds[i].item()]
                predicted_chars.append(char)
                
            predicted_text = ''.join(predicted_chars)
            return predicted_text
            
        except Exception as e:
            print(f"Error predicting CAPTCHA: {e}")
            return None

# Global instance for easy access
recognizer = None

def recognize_captcha_with_model(image_path: str) -> Optional[str]:
    """
    Recognize CAPTCHA using the trained model (drop-in replacement for Gemini service)
    
    Args:
        image_path (str): Path to the CAPTCHA image file
        
    Returns:
        str: Recognized CAPTCHA text or None if failed
    """
    global recognizer
    
    # Initialize recognizer on first use
    if recognizer is None:
        try:
            print(f"Initializing CAPTCHA recognizer...")
            print(f"Project root: {project_root}")
            print(f"Model path: {model_path}")
            print(f"Train labels path: {train_labels_path}")
            recognizer = CaptchaModelRecognizer()
        except Exception as e:
            print(f"Failed to initialize CAPTCHA recognizer: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Predict CAPTCHA text
    try:
        captcha_text = recognizer.predict(image_path)
        if captcha_text and len(captcha_text) == 6:
            print(f"🤖 Model recognized CAPTCHA as: {captcha_text}")
            return captcha_text
        else:
            print(f"Invalid CAPTCHA prediction: '{captcha_text}' (length: {len(captcha_text) if captcha_text else 0})")
            return None
    except Exception as e:
        print(f"Error recognizing CAPTCHA with model: {e}")
        return None

if __name__ == "__main__":
    # Test the recognizer
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            result = recognize_captcha_with_model(image_path)
            print(f"Prediction: {result}")
        else:
            print(f"Image not found: {image_path}")
    else:
        print("Usage: python captcha_model_recognizer.py <captcha_image_path>")