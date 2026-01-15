
import torch
import sys
import os
from PIL import Image
from torchvision import transforms

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.model import get_model
from src.poison_detector import PoisonDetector

def check_error():
    print("Checking reconstruction error...")
    device = Config.DEVICE
    
    # Load Detector
    feature_extractor = get_model(pretrained=True)
    detector = PoisonDetector(feature_extractor)
    detector.load_detector("poison_detector.pth")
    detector.detector.eval()
    feature_extractor.eval()
    
    # Load Image
    img_path = os.path.join("poison_samples", "normal_attacked.png")
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return

    transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    
    # Get Error
    with torch.no_grad():
        features = detector.extract_features(tensor)
        reconstructed = detector.detector(features)
        error = torch.mean((features - reconstructed) ** 2).item()
        
    with open("debug_result.txt", "w") as f:
        f.write(f"Reconstruction Error for attacked image: {error:.6f}\n")
        f.write(f"Current Threshold: {detector.threshold:.6f}\n")
        
        # Also check clean image
        clean_path = os.path.join("poison_samples", "normal_clean.png")
        if os.path.exists(clean_path):
            img_clean = Image.open(clean_path).convert('RGB')
            tensor_clean = transform(img_clean).unsqueeze(0).to(device)
            with torch.no_grad():
                features_clean = detector.extract_features(tensor_clean)
                reconstructed_clean = detector.detector(features_clean)
                error_clean = torch.mean((features_clean - reconstructed_clean) ** 2).item()
            f.write(f"Reconstruction Error for CLEAN image: {error_clean:.6f}\n")

if __name__ == "__main__":
    check_error()
