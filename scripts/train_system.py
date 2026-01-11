import torch
import numpy as np
import random
import os
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import Config
from src.data_loader import create_federated_datasets, get_dataloaders
from src.model import get_model
from src.adversarial_attacks import AdversarialAttacks
from src.poison_detector import PoisonDetector
from src.federated_learning import FederatedLearning
from torch.utils.data import DataLoader

def set_seed(seed=Config.RANDOM_SEED):
    """Fixer les seeds pour la reproductibilité"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def print_header(text):
    """Afficher un en-tête formaté"""
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}\n")

def main():
    # Configuration initiale
    set_seed()
    
    print_header("🚀 SYSTÈME DE DÉTECTION D'ATTAQUES ADVERSARIALES")
    print(f"Device utilisé: {Config.DEVICE}")
    print(f"GPU disponible: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Mémoire GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Étape 1: Charger les données
    print_header("📁 ÉTAPE 1: CHARGEMENT DES DONNÉES")
    
    if not os.path.exists(Config.DATASET_PATH):
        print("❌ Dataset non trouvé!")
        print("Exécutez d'abord: python download_data.py")
        return
    
    # Créer les datasets fédérés
    hospital_datasets = create_federated_datasets(Config.DATASET_PATH)
    print(f"✓ {Config.NUM_HOSPITALS} hôpitaux créés")
    for i, dataset in enumerate(hospital_datasets):
        print(f"  - Hôpital {i+1}: {len(dataset)} images")
    
    # Charger les données de test
    test_loader, val_loader = get_dataloaders(Config.DATASET_PATH)
    print(f"✓ Dataset de test: {len(test_loader.dataset)} images")
    print(f"✓ Dataset de validation: {len(val_loader.dataset)} images")
    
    # Étape 2: Pré-entraînement du modèle
    print_header("🧠 ÉTAPE 2: PRÉ-ENTRAÎNEMENT DU MODÈLE")
    
    pretrained_model = get_model(pretrained=True)
    print("✓ Modèle ResNet18 pré-entraîné chargé")
    
    # Étape 3: Génération d'attaques adversariales (POUR LE TEST SEULEMENT)
    print_header("⚔️ ÉTAPE 3: GÉNÉRATION D'ATTAQUES (SIMULATION D'ATTAQUE)")
    
    print("Note: Les attaques ne sont plus utilisées pour entraîner le détecteur (Unsupervised).")
    print("Elles serviront uniquement à tester la robustesse et simuler une attaque sur l'Hôpital 2.")
    
    # Utiliser le premier dataset d'hôpital pour l'entraînement propre
    sample_loader = DataLoader(
        hospital_datasets[0],
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )
    
    # Étape 4: Entraînement du détecteur d'attaques (AUTOENCODER)
    print_header("🔍 ÉTAPE 4: ENTRAÎNEMENT DU DÉTECTEUR (AUTOENCODER)")
    
    print("Entraînement sur des données PROPRES uniquement...")
    poison_detector = PoisonDetector(pretrained_model)
    
    # Entraîner sur les données propres de l'Hôpital 0
    poison_detector.train_detector(sample_loader, epochs=20)
    poison_detector.save_detector("poison_detector.pth")
    
    # Étape 5: Filtrage global des données (Gatekeeper)
    print_header("🧹 ÉTAPE 5: FILTRAGE DE TOUS LES HÔPITAUX")
    
    print("Utilisation de l'Autoencoder pour nettoyer les données de CHAQUE hôpital avant l'apprentissage fédéré...")
    
    # Simuler une attaque sur l'Hôpital 2 pour prouver que ça marche
    print("\n[SIMULATION] Injection d'attaques dans l'Hôpital 2 pour tester le filtre...")
    poisoned_loader = DataLoader(hospital_datasets[1], batch_size=Config.BATCH_SIZE, shuffle=False)
    attacked_data = AdversarialAttacks.generate_adversarial_dataset(
        pretrained_model, poisoned_loader, attack_type='pgd', ratio=0.5
    )
    # Créer le dataset attaqué
    from torch.utils.data import TensorDataset
    att_img = torch.stack([i[0] for i in attacked_data])
    att_lbl = torch.as_tensor([i[1] for i in attacked_data])
    hospital_datasets[1] = TensorDataset(att_img, att_lbl)
    print(f"⚠️ Hôpital 2 corrompu ! (Contient maintenant {len(hospital_datasets[1])} images mixtes)")

    # Boucle de nettoyage sur TOUS les hôpitaux
    for i in range(len(hospital_datasets)):
        print(f"\n🏥 Nettoyage Hôpital {i+1}...")
        
        # 1. Créer loader
        current_loader = DataLoader(
            hospital_datasets[i], 
            batch_size=Config.BATCH_SIZE, 
            shuffle=False,
            num_workers=2
        )
        
        # 2. Filtrer
        clean_data = poison_detector.filter_clean_data(current_loader)
        
        # 3. Mettre à jour le dataset
        if len(clean_data) > 0:
            clean_images = torch.stack([item[0] for item in clean_data])
            clean_labels = torch.stack([item[1] for item in clean_data])
            hospital_datasets[i] = TensorDataset(clean_images, clean_labels)
            print(f"✓ Hôpital {i+1} validé: {len(hospital_datasets[i])} images propres prêtes pour FL.")
        else:
            print(f"⚠️ Hôpital {i+1}: Toutes les données ont été rejetées ! (Mode Paranoiaque ?)")
    
    # Étape 6: Apprentissage fédéré avec données propres
    print_header("🏥 ÉTAPE 6: APPRENTISSAGE FÉDÉRÉ")
    
    # Créer un nouveau modèle global
    global_model = get_model(pretrained=True)
    
    # Initialiser l'apprentissage fédéré
    fed_learning = FederatedLearning(global_model)
    
    # Entraîner de manière fédérée
    final_model = fed_learning.federated_training(hospital_datasets)
    
    # Étape 7: Évaluation finale
    print_header("📊 ÉTAPE 7: ÉVALUATION FINALE")
    
    # Évaluer le modèle global
    accuracy = fed_learning.evaluate_global_model(test_loader)
    
    # Tester la robustesse contre les attaques
    print("\n🛡️ Test de robustesse contre les attaques...")
    
    # Générer des exemples adversariaux sur le test set
    test_adv_fgsm = []
    test_adv_pgd = []
    
    for images, labels in test_loader:
        images = images.to(Config.DEVICE)
        labels = labels.to(Config.DEVICE)
        
        # FGSM
        adv_fgsm = AdversarialAttacks.fgsm_attack(final_model, images, labels)
        test_adv_fgsm.append((adv_fgsm, labels))
        
        # PGD
        adv_pgd = AdversarialAttacks.pgd_attack(final_model, images, labels)
        test_adv_pgd.append((adv_pgd, labels))
    
    # Évaluer sur les données adversariales
    print("\nÉvaluation sur données originales:")
    print(f"  Accuracy: {accuracy:.2f}%")
    
    # Sauvegarder les modèles
    print_header("💾 SAUVEGARDE DES MODÈLES")
    fed_learning.save_global_model("global_model_final.pth")
    
    print_header("✅ PROCESSUS TERMINÉ AVEC SUCCÈS")
    print("Fichiers générés:")
    print("  - poison_detector.pth")
    print("  - global_model_final.pth")
    print("\nVous pouvez maintenant utiliser ces modèles pour:")
    print("  1. Détecter les attaques adversariales")
    print("  2. Classifier les radiographies thoraciques")
    print("  3. Poursuivre l'entraînement fédéré")

if __name__ == "__main__":
    main()