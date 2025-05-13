import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import argparse
from cnn_model import VehicleCNN
from metrics import TrainingMetrics
from torchvision import datasets, transforms
import os

def create_dummy_data(num_classes, batch_size):
    """Create dummy data for testing when real data isn't available"""
    images = torch.randn(1000, 3, 224, 224)
    labels = torch.randint(0, num_classes, (1000,))
    
    dataset = TensorDataset(images, labels)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def load_real_data(data_dir, batch_size):
    """Load real vehicle classification data from the processed folder"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'test'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, metrics):
    model.train()
    metrics.reset()
    
    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}', leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        metrics.update(loss, outputs, targets)
        progress_bar.set_postfix({'Loss': f"{loss.item():.4f}"})
    
    return metrics.compute()

def validate(model, dataloader, criterion, device, metrics):
    model.eval()
    metrics.reset()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metrics.update(loss, outputs, targets)
    
    return metrics.compute()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    
    model = VehicleCNN(num_classes=args.num_classes).to(device)
    
    # 📌 Utilise l’une des deux lignes ci-dessous selon le besoin
    # train_loader, val_loader = create_dummy_data(args.num_classes, args.batch_size)
    train_loader, val_loader = load_real_data('data/processed', args.batch_size)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_metrics = TrainingMetrics(args.num_classes)
    val_metrics = TrainingMetrics(args.num_classes)
    
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, train_metrics)
        val_stats = validate(model, val_loader, criterion, device, val_metrics)
        
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print(f"Train - Loss: {train_stats['loss']:.4f}, Acc: {train_stats['accuracy']:.2f}%")
        print(f"Val   - Loss: {val_stats['loss']:.4f}, Acc: {val_stats['accuracy']:.2f}%")
        
        scheduler.step()
        
        if epoch % args.save_interval == 0:
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"checkpoints/checkpoint_epoch{epoch}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--num-classes', type=int, default=10)
    parser.add_argument('--save-interval', type=int, default=5)
    
    args = parser.parse_args()
    
    main(args)
