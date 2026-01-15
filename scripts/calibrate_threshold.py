import sys
import os
import torch
import numpy as np
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.data_loader import get_dataloaders
from src.model import get_model
from src.poison_detector import PoisonDetector

def calibrate():
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    # Load Data
    print("Loading test data for calibration...")
    # Using 'test' loader as a proxy for clean data validation
    test_loader, _ = get_dataloaders(Config.DATASET_PATH) 
    
    # Load Model
    print("Loading Feature Extractor...")
    model = get_model(pretrained=False)
    # We assume model weights are not strictly necessary for feature shape, 
    # but good to have if we want meaningful features. 
    # Attempt to load if exists, else random (might be bad if not same as training)
    model_path = "global_model_final.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    
    model.to(device)
    model.eval()

    # Load Detector
    print("Loading Poison Detector...")
    detector = PoisonDetector(model)
    detector_path = "poison_detector.pth"
    if os.path.exists(detector_path):
        detector.load_detector(detector_path)
    else:
        print("Warning: No poison_detector.pth found. Using uninitialized detector.")
    
    detector.detector.to(device)
    detector.detector.eval()
    
    # Calculate Errors on Clean Data
    print("Calculating reconstruction errors on clean data...")
    errors = []
    
    with torch.no_grad():
        for images, _ in tqdm(test_loader):
            images = images.to(device)
            # detect_poison returns (is_poisoned, errors_tensor)
            _, batch_errors = detector.detect_poison(images)
            errors.extend(batch_errors.cpu().numpy())
            
    errors = np.array(errors)
    mean_err = np.mean(errors)
    std_err = np.std(errors)
    max_err = np.max(errors)
    min_err = np.min(errors)
    
    print(f"\nStats for Clean Data ({len(errors)} samples):")
    print(f"  Min: {min_err:.6f}")
    print(f"  Max: {max_err:.6f}")
    print(f"  Mean: {mean_err:.6f}")
    print(f"  Std: {std_err:.6f}")
    
    # Heuristic: Mean + 3*Std
    suggested_threshold = mean_err + 3 * std_err
    print(f"\nSuggested Threshold (Mean + 3*Std): {suggested_threshold:.6f}")
    
    return suggested_threshold

if __name__ == "__main__":
    calibrate()
