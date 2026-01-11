import torch
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.data_loader import create_federated_datasets, get_dataloaders
from src.model import get_model, get_poison_detector
from src.poison_detector import PoisonDetector
from src.federated_learning import FederatedLearning
import os

def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")

def main():
    print_header("🏥 APPRENTISSAGE FÉDÉRÉ UNIQUEMENT (FL ONLY)")
    
    # 1. Chargement des données
    print("Chargement des datasets...")
    if not os.path.exists(Config.DATASET_PATH):
        print("❌ Dataset non trouvé! Veuillez exécuter l'option 1 du menu.")
        return

    hospital_datasets = create_federated_datasets(Config.DATASET_PATH)
    test_loader, _ = get_dataloaders(Config.DATASET_PATH)
    
    # 2. Chargement du modèle
    print("Initialisation du modèle Global (EfficientNet-V2)...")
    global_model = get_model(pretrained=True)
    
    # 3. Chargement du Détecteur (Autoencoder) existant
    print("\n🔍 Verification du détecteur de poison...")
    if os.path.exists("poison_detector.pth"):
        print("✓ Détecteur trouvé ('poison_detector.pth'). Chargement...")
        poison_detector = PoisonDetector(global_model)
        poison_detector.load_detector("poison_detector.pth")
        
        # 4. Filtrage (Gatekeeper)
        print_header("🧹 PRE-FILTRAGE DE TOUS LES HÔPITAUX")
        print("Nettoyage des données avant FL...")
        
        for i in range(len(hospital_datasets)):
            print(f"Traîtement Hôpital {i+1}...")
            loader = DataLoader(hospital_datasets[i], batch_size=Config.BATCH_SIZE, shuffle=False)
            
            # Utilisation du détecteur
            # Note: filter_clean_data utilise le threshold calibré chargé
            clean_data = poison_detector.filter_clean_data(loader)
            
            if len(clean_data) > 0:
                clean_imgs = torch.stack([x[0] for x in clean_data])
                clean_lbls = torch.stack([x[1] for x in clean_data])
                hospital_datasets[i] = TensorDataset(clean_imgs, clean_lbls)
                print(f"  ✓ Hôpital {i+1}: {len(hospital_datasets[i])} images valides.")
            else:
                print(f"  ⚠️ Hôpital {i+1}: VIDE (Tout rejeté)")
                
    else:
        print("⚠️ 'poison_detector.pth' NON TROUVÉ.")
        print("⚠️ ATTENTION: L'apprentissage fédéré va démarrer SANS filtrage de sécurité.")
        print("   (Pour activer la sécurité, lancez l'entraînement complet une fois)")
        
    # 5. Apprentissage Fédéré
    print_header("🚀 DÉMARRAGE DE L'APPRENTISSAGE FÉDÉRÉ")
    fl_system = FederatedLearning(global_model)
    
    # Exécution
    global_model = fl_system.federated_training(hospital_datasets)
    
    # 6. Évaluation
    print_header("📊 ÉVALUATION FINALE")
    accuracy = fl_system.evaluate_global_model(test_loader)
    
    # 7. Sauvegarde
    fl_system.save_global_model("global_model_final.pth")
    print("\n✓ Terminé.")

if __name__ == "__main__":
    main()
