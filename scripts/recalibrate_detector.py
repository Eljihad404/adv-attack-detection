
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.model import get_model
from src.poison_detector import PoisonDetector
from src.data_loader import get_transforms

def recalibrate():
    print("🚀 Recalibration du détecteur d'anomalies...")
    
    # 1. Setup
    device = Config.DEVICE
    batch_size = 16
    epochs = 30 # Requested by user
    lr = 0.001
    
    # 2. Data Loader (ALL Clean Data)
    _, test_transform = get_transforms()
    
    datasets_list = []
    for split in ["train", "val", "test"]:
        path = os.path.join(Config.DATASET_PATH, split)
        if os.path.exists(path):
            print(f"➕ Ajout du dataset: {split}")
            datasets_list.append(datasets.ImageFolder(root=path, transform=test_transform))
        else:
            print(f"⚠️ {split} introuvable, ignoré.")
            
    if not datasets_list:
        print("❌ Aucune donnée trouvée.")
        return
        
    full_dataset = torch.utils.data.ConcatDataset(datasets_list)
    dataloader = DataLoader(full_dataset, batch_size=batch_size, shuffle=True)
    
    print(f"📊 Données d'entraînement (TOTAL): {len(full_dataset)} images")
    
    # 3. Initialize Models
    # Backbone (ImageNet - Fixed)
    backbone = get_model(pretrained=True)
    backbone.eval()
    
    # Detector (Autoencoder - To be trained)
    # Start fresh or finetune? Let's start fresh to ensure clean slate
    poison_detector = PoisonDetector(backbone)
    
    # 4. Training Loop
    print("\n🔄 Début de l'entraînement...")
    poison_detector.train_detector(dataloader, epochs=epochs)
    
    # 5. Save
    print("\n💾 Sauvegarde du nouveau détecteur...")
    poison_detector.save_detector("poison_detector.pth")
    
    print("\n✅ Recalibration terminée!")

if __name__ == "__main__":
    recalibrate()
