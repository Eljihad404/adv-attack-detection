import torch

class Config:
    # -------------------------
    # Données
    # -------------------------
    DATASET_PATH = "chest_xray"
    CLEAN_DATA_PATH = "clean_data"

    # -------------------------
    # Modèle de classification
    # -------------------------
    NUM_CLASSES = 2
    IMG_SIZE = 224
    BATCH_SIZE = 32       # OK pour RTX 4060 8GB
    EPOCHS = 12
    LEARNING_RATE = 1e-4 # Reduced from 1e-3 for better fine-tuning

    # -------------------------
    # Fédéré (optionnel)
    # -------------------------
    NUM_HOSPITALS = 4
    FEDERATED_ROUNDS = 15 # Increased from 5 for better convergence

    # -------------------------
    # Attaques adversariales
    # -------------------------
    EPSILON_FGSM = 0.03
    EPSILON_PGD = 0.03
    PGD_ALPHA = 0.01
    PGD_ITERATIONS = 10

    # -------------------------
    # Détection adversariale
    # -------------------------
    DETECTION_THRESHOLD = 0.018  # Recalibrated based on observed error (Clean: ~0.014, Fast Attack: ~0.024)

    # -------------------------
    # Matériel
    # -------------------------
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 0 # Set to 0 for Windows compatibility/Debugging

    # -------------------------
    # Reproductibilité
    # -------------------------
    RANDOM_SEED = 42


    DETECTOR_LR = 0.001