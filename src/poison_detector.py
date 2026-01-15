import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from .config import Config
from .model import get_poison_detector

class PoisonDetector:
    """
    Détecteur d'attaques adversariales basé sur un Autoencoder Résiduel.
    Utilise la détection d'anomalies: les attaques ont une erreur de reconstruction plus élevée.
    """
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.detector = get_poison_detector() # ResidualAutoencoder
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.detector.parameters(), lr=0.001)
        
        # Statistiques pour le seuil dynamique
        self.threshold = 0.0
        self.mean_error = 0.0
        self.std_error = 0.0
    
    def extract_features(self, images):
        """Extraire les features des images"""
        self.feature_extractor.eval()
        with torch.no_grad():
            features = self.feature_extractor.get_features(images)
        return features
    
    def train_detector(self, train_loader, epochs=10):
        """
        Entraîner l'Autoencoder sur des données PROPRES uniquement (Non-supervisé).
        
        Args:
            train_loader: DataLoader contenant uniquement des images propres
            epochs: Nombre d'époques
        """
        print("\n🔍 Entraînement de l'Autoencoder (Détection d'anomalies)...")
        
        self.detector.train()
        
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for images, _ in progress_bar: # On ignore les labels
                images = images.to(Config.DEVICE)
                
                # 1. Extraire les features (Fixed backbone)
                with torch.no_grad():
                    features = self.extract_features(images)
                
                # 2. Forward pass (Autoencoder)
                reconstructed = self.detector(features)
                
                # 3. Loss = Reconstruction Error (MSE)
                loss = self.criterion(reconstructed, features)
                
                # 4. Backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
            
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}: Avg Reconstruction Error = {avg_loss:.6f}")
        
        # Calibrer le seuil après l'entraînement
        self._calibrate_threshold(train_loader)
            
    def _calibrate_threshold(self, dataloader):
        """Calculer la moyenne et l'écart-type des erreurs de reconstruction sur les données propres"""
        print("\n📏 Calibration du seuil de détection...")
        self.detector.eval()
        errors = []
        
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="Calibration"):
                images = images.to(Config.DEVICE)
                features = self.extract_features(images)
                reconstructed = self.detector(features)
                
                # Erreur par sample (MSE reduction='none')
                loss = torch.mean((features - reconstructed) ** 2, dim=1)
                errors.extend(loss.cpu().numpy())
        
        errors = np.array(errors)
        self.mean_error = float(np.mean(errors))
        self.std_error = float(np.std(errors))
        
        # Règle empirique: seuil = moyenne + 3 * ecart-type (99.7% confidence)
        self.threshold = self.mean_error + 3 * self.std_error
        
        print(f"✓ Calibration terminée:")
        print(f"  - Mean Error: {self.mean_error:.6f}")
        print(f"  - Std Dev: {self.std_error:.6f}")
        print(f"  - Seuil défini: {self.threshold:.6f} (Mean + 3*Std)")

    def detect_poison(self, images):
        """
        Détecter si des images sont adversariales (anomalies)
        
        Returns:
            is_poisoned: Bool tensor
            errors: Reconstruction errors
        """
        self.detector.eval()
        
        with torch.no_grad():
            features = self.extract_features(images)
            reconstructed = self.detector(features)
            
            # Calculer MSE par sample
            errors = torch.mean((features - reconstructed) ** 2, dim=1)
            
            # Si erreur > seuil => Poison
            is_poisoned = errors > self.threshold
        
        return is_poisoned, errors
    
    def filter_clean_data(self, dataloader, output_path=Config.CLEAN_DATA_PATH):
        """Filtrer les données en rejetant celles avec une haute erreur de reconstruction"""
        print(f"\n🧹 Filtrage basé sur l'erreur de reconstruction (Seuil: {self.threshold:.6f})...")
        
        clean_data = []
        total_images = 0
        poisoned_images = 0
        
        self.detector.eval()
        
        for images, labels in tqdm(dataloader, desc="Filtrage"):
            images = images.to(Config.DEVICE)
            
            is_poisoned, errors = self.detect_poison(images)
            
            for i in range(len(images)):
                total_images += 1
                if not is_poisoned[i]:
                    clean_data.append((images[i].cpu(), labels[i].cpu()))
                else:
                    poisoned_images += 1
        
        detection_rate = 100. * poisoned_images / total_images
        print(f"\n✓ Filtrage terminé:")
        print(f"  - Total: {total_images}")
        print(f"  - Rejetés (Poison/Anomalie): {poisoned_images} ({detection_rate:.2f}%)")
        print(f"  - Gardés (Propres): {len(clean_data)}")
        
        return clean_data
    
    def save_detector(self, path="poison_detector.pth"):
        torch.save({
            'state_dict': self.detector.state_dict(),
            'threshold': self.threshold,
            'mean_error': self.mean_error,
            'std_error': self.std_error
        }, path)
        print(f"✓ Autoencoder sauvegardé dans {path}")
    
    def load_detector(self, path="poison_detector.pth"):
        checkpoint = torch.load(path, map_location=Config.DEVICE)
        
        state_dict = None
        # Gestion flexible des clés
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'detector_state_dict' in checkpoint:
            print("ℹ️ Chargement via clé 'detector_state_dict'")
            state_dict = checkpoint['detector_state_dict']
        else:
            print("⚠️ Aucune clé de dictionnaire d'état connue trouvée (state_dict ou detector_state_dict)")
            # On essaie de charger le checkpoint directement si c'est juste le state_dict
            state_dict = checkpoint

        if state_dict:
            # Correction des clés préfixées par 'detector.'
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('detector.'):
                    new_key = k.replace('detector.', '', 1)
                    new_state_dict[new_key] = v
                else:
                    new_state_dict[k] = v
            
            try:
                self.detector.load_state_dict(new_state_dict)
            except Exception as e:
                print(f"❌ Erreur lors du chargement des poids: {e}")
                # Fallback: essai brutal, peut-être que les clés étaient bonnes ?
                # self.detector.load_state_dict(state_dict)
        
        # Override with Config value (Manual Correction)
        self.threshold = Config.DETECTION_THRESHOLD # Enforce Config value (0.62)
        
        self.mean_error = checkpoint.get('mean_error', 0.0)
        self.std_error = checkpoint.get('std_error', 0.0)
        print(f"✓ Autoencoder chargé (Seuil CORRIGÉ: {self.threshold:.6f})")