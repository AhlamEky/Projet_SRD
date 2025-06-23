import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
from cnn_model import VehicleCNN

# 1. Lecture des arguments
parser = argparse.ArgumentParser()
parser.add_argument('--rank', type=int, required=True, help="Identifiant du noeud (0 = master)")
parser.add_argument('--world_size', type=int, required=True, help="Nombre total de machines")
parser.add_argument('--epochs', type=int, default=10, help="Nombre d'époques d'entraînement")
args = parser.parse_args()

# 2. Initialisation du processus distribué
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = '100.123.244.20'  # IP Tailscale correcte (Ahlam)
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(
      backend="gloo",
      init_method="tcp://100.123.244.20:12355",
      rank=rank,
      world_size=world_size
    )  

def cleanup():
    dist.destroy_process_group()

# 3. Fonction d'entraînement principale
def train(rank, world_size, epochs):
    setup(rank, world_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prétraitement
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(root='data/processed/train', transform=transform)
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=32, sampler=train_sampler)

    model = VehicleCNN().to(device)
    ddp_model = DDP(model)

    optimizer = optim.Adam(ddp_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        ddp_model.train()
        train_sampler.set_epoch(epoch)
        running_loss = 0.0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = ddp_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"[Rank {rank}] Epoch {epoch+1}/{epochs} - Loss: {running_loss:.4f}")

    # Sauvegarde du modèle (uniquement sur le maître)
    if rank == 0:
        torch.save(ddp_model.module.state_dict(), "vehicle_model_distributed.pth")
        print("[✅ Master] Modèle sauvegardé : vehicle_model_distributed.pth")

    cleanup()

# 4. Exécution
if __name__ == "__main__":
    train(args.rank, args.world_size, args.epochs)
