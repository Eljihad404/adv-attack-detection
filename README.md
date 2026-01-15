# 🛡️ Federated Adversarial Attack Detection System for Medical Imaging

A comprehensive Federated Learning system designed to detect adversarial attacks (FGSM and PGD) on Chest X-Ray images. This project simulates a multi-hospital environment where a global model is trained securely while being protected by a dedicated Poison Autoencoder Detector.

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Key Features](#-key-features)
- [Prerequisites](#-prerequisites)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
    - [1. Data Setup](#1-data-setup)
    - [2. System Training](#2-system-training)
    - [3. Running the Application](#3-running-the-application)
- [Configuration](#%EF%B8%8F-configuration)
- [Results](#-expected-results)

## 🏗️ Architecture

The system implements a robust pipeline ensuring data integrity before federated aggregation:

```mermaid
graph TD
    A[Hospitals (N Datasets)] --> B[Poison Detection (Autoencoder)]
    B -->|Clean Data| C[Federated Learning (FedAvg)]
    B -->|Poisoned Data| D[Discarded]
    C --> E[Global Model (EfficientNetV2)]
    E --> F[Deployment (API + Frontend)]
```

### Components:
1.  **Multi-Source Simulation**: Simulates N hospitals (default: 4) with independent datasets.
2.  **Poison Detector**: A Residual Autoencoder trained on clean data to flag anomalies (high reconstruction error).
3.  **Adversarial Attacks**: Implementation of FGSM and PGD for robustness testing.
4.  **Federated Learning**: Uses the FedAvg algorithm to aggregate local model updates into a global EfficientNetV2-S model.
5.  **Modern Interface**: A FastAPI backend coupled with a React/Vite frontend for real-time inference.

## ✨ Key Features

*   **Model**: **EfficientNetV2-S** (Pretrained) modified for binary classification (Normal vs Pneumonia).
*   **Defense**: Unsupervised Anomaly Detection using a custom **Residual Autoencoder**.
*   **Attacks**: Robustness tested against FGSM and PGD adversarial samples.
*   **Privacy**: Federated Learning ensures raw patient data never leaves the local "hospital" scope.

## 🔧 Prerequisites

*   **OS**: Windows 10/11 (Optimized for)
*   **Hardware**: GPU recommended (Tested on RTX 4060 8GB).
*   **Python**: 3.8+
*   **Node.js**: 16+ (For the frontend)
*   **Kaggle Account**: To download the Chest X-Ray dataset.

## 📁 Project Structure

```
project/
├── api/
│   └── server.py             # FastAPI Backend
├── frontend/                 # React + Vite Web Application
├── scripts/
│   ├── download_data.py      # Dataset downloader
│   ├── train_system.py       # Main full-system training script
│   └── train_fl.py           # Federated Learning exclusive script
├── src/
│   ├── config.py             # Global Configuration parameters
│   ├── model.py              # EfficientNetV2 & Autoencoder definitions
│   ├── federated_learning.py # FL Logic (FedAvg)
│   ├── poison_detector.py    # Detector logic
│   └── ...
├── requirements.txt          # Python dependencies
└── README.md                 # Project Documentation
```

## 📥 Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd <repository-name>
```

### 2. Backend Setup
Create a virtual environment and install dependencies:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Install deps
pip install -r requirements.txt
```

### 3. Frontend Setup
Install Node.js dependencies:
```bash
cd frontend
npm install
cd ..
```

### 4. Kaggle Configuration
1.  Place your `kaggle.json` API token in `C:\Users\<YourUser>\.kaggle\`.
2.  Accept the dataset rules at [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia).

## 🚀 Usage

### 1. Data Setup
Download the dataset using the script:
```bash
python scripts/download_data.py
```

### 2. System Training
Run the complete pipeline (Data Prep -> Detector Training -> Attack Simulation -> Federated Learning):
```bash
python scripts/train_system.py
```
This process will:
*   Train the **Poison Detector** (Autoencoder) on clean data.
*   Simulate an attack on one hospital (e.g., Hospital 2).
*   Filter poisoned data using the Detector.
*   Train the **Global Model** using Federated Learning (15 Rounds by default).
*   Save `poison_detector.pth` and `global_model_final.pth`.

### 3. Running the Application
Start the full stack application (Backend + Frontend).

**Terminal 1 (Backend):**
```bash
python api/server.py
```
*Server runs at `http://localhost:8000`*

**Terminal 2 (Frontend):**
```bash
cd frontend
npm run dev
```
*App runs at `http://localhost:5173` (typically)*

Open your browser to the frontend URL to use the interface.

## ⚙️ Configuration

You can tune the system parameters in `src/config.py`:

```python
# Model
BATCH_SIZE = 32           # Adjust based on GPU VRAM
EPOCHS = 12               
LEARNING_RATE = 1e-4

# Federated Learning
NUM_HOSPITALS = 4
FEDERATED_ROUNDS = 15     # Number of aggregation rounds

# Attacks
EPSILON_FGSM = 0.03
EPSILON_PGD = 0.03

# Detector
DETECTION_THRESHOLD = 0.018  # Reconstruction error threshold
```

## 📊 Expected Results

*   **Global Model Accuracy**: Aiming for >85% on clean test data.
*   **Detector Efficiency**: Should filter out >80% of adversarial samples while keeping False Positives low.
*   **Files**: Training generates `global_model_final.pth` (~80MB) and `poison_detector.pth`.

## 🤝 Contribution

Contributions are welcome! Please fork the repository and submit a Pull Request.

## 📄 License

Educational purpose only. Dataset subject to Kaggle's license.