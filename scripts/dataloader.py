import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms

def create_dataloaders(processed_dir, transform=None, batch_size=64):
    """Crée des DataLoaders pour train/val/test"""
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    datasets = {
        split: ImageFolder(root=f"{processed_dir}/{split}", transform=transform)
        for split in ['train', 'val', 'test']
    }
    
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(datasets['val'], batch_size=batch_size, shuffle=False, num_workers=4),
        'test': DataLoader(datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    
    return dataloaders