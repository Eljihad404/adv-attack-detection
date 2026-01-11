import torch
import torch.nn as nn
import torchvision.models as models
from .config import Config

class ChestXRayModel(nn.Module):
    """
    Modèle basé sur EfficientNet-V2-S pour la classification de radiographies thoraciques
    Optimisé pour RTX 4060 8GB
    """
    def __init__(self, num_classes=Config.NUM_CLASSES, pretrained=True):
        super(ChestXRayModel, self).__init__()
        
        # Utiliser EfficientNet-V2-S
        weights = models.EfficientNet_V2_S_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_v2_s(weights=weights)
        
        # Remplacer la dernière couche (classifier) pour notre tâche
        # EfficientNet a un classifier qui est un Sequential(Dropout, Linear)
        # Input features pour V2-S est 1280
        num_features = self.backbone.classifier[1].in_features
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """Extraire les features avant la classification"""
        # Pour EfficientNet, on passe par les features extraction layers
        x = self.backbone.features(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        
        return x

class ResidualBlock(nn.Module):
    """
    Bloc Résiduel pour l'Autoencoder
    Permet un flux de gradient plus profond et une meilleure reconstruction
    """
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim)
        )
        self.activation = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return self.activation(out)

class PoisonDetectionModel(nn.Module):
    """
    Autoencoder Résiduel pour la détection d'anomalies (attaques)
    Entraîné pour reconstruire les données propres.
    Une erreur de reconstruction élevée indique une attaque.
    """
    def __init__(self, input_dim=1280):
        super(PoisonDetectionModel, self).__init__()
        
        # Encoder: Compression vers l'espace latent
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Innovation: Ajout d'un bloc résiduel
            ResidualBlock(512),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 128) # Latent Space
        )
        
        # Decoder: Reconstruction depuis l'espace latent
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # Innovation: Bloc résiduel dans le décaleur aussi
            ResidualBlock(512),
            
            nn.Linear(512, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def get_model(pretrained=True):
    """Créer et retourner le modèle"""
    model = ChestXRayModel(pretrained=pretrained)
    return model.to(Config.DEVICE)

def get_poison_detector(input_dim=1280):
    """Créer et retourner le détecteur de poison"""
    detector = PoisonDetectionModel(input_dim=input_dim)
    return detector.to(Config.DEVICE)