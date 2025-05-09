import os
import shutil
import random
from tqdm import tqdm
from PIL import Image
from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# ============================
# CONFIGURATION GLOBALE
# ============================

BASE_DIR = Path(__file__).parent.parent  # Racine du projet
RAW_DATA_DIR = BASE_DIR / "data/raw"     # Chemin ABSOLU
PROCESSED_DATA_DIR = BASE_DIR / "data/processed"  
IMAGE_SIZE = (224, 224)
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
SEED = 42

# Vérifier si un GPU est disponible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================
#  TRANSFORMATIONS
# ============================
transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(),  # Augmentation aléatoire des images
    transforms.RandomRotation(20),      # Rotation aléatoire
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation [-1, 1]
])

# ============================
# FONCTION : CRÉATION DOSSIERS
# ============================
def create_dirs():
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(PROCESSED_DATA_DIR, split)
        os.makedirs(split_dir, exist_ok=True)

# ============================
# DIVISER LES DONNÉES
# ============================
def split_dataset():
    random.seed(SEED)
    classes = os.listdir(RAW_DATA_DIR)

    for class_name in tqdm(classes, desc="Prétraitement"):
        class_path = os.path.join(RAW_DATA_DIR, class_name)
        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)
        random.shuffle(images)

        total = len(images)
        train_end = int(TRAIN_SPLIT * total)
        val_end = train_end + int(VAL_SPLIT * total)

        splits = {
            'train': images[:train_end],
            'val': images[train_end:val_end],
            'test': images[val_end:]
        }

        for split, split_images in splits.items():
            dest_dir = os.path.join(PROCESSED_DATA_DIR, split, class_name)
            os.makedirs(dest_dir, exist_ok=True)

            for img_name in split_images:
                src_path = os.path.join(class_path, img_name)
                dest_path = os.path.join(dest_dir, img_name)
                try:
                    img = Image.open(src_path).convert('RGB')
                    img = transform(img)  # Transformer l'image en Tensor
                    img = img.to(device)  # Déplacer l'image sur le GPU si disponible
                    
                    # Sauvegarder l'image après transformation
                    save_img = transforms.ToPILImage()(img.cpu())  # Revenir sur le CPU pour la sauvegarde
                    save_img.save(dest_path)
                except Exception as e:
                    print(f" Erreur avec {img_name} : {e}")
                    continue

# ============================
def run_preprocessing():
    print("Création des dossiers de sortie...")
    create_dirs()
    print("Division et traitement des images en cours...")
    split_dataset()
    print("Prétraitement terminé.")

# ============================
if __name__ == "__main__":
    run_preprocessing()