import torch
from sklearn.metrics import precision_score, recall_score, f1_score

class TrainingMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        self.loss = 0
        self.correct = 0
        self.total = 0
        self.all_preds = []
        self.all_targets = []
    
    def update(self, loss, preds, targets):
        self.loss += loss.item()
        _, predicted = torch.max(preds.data, 1)
        self.total += targets.size(0)
        self.correct += (predicted == targets).sum().item()
        
        # Stockage pour métriques supplémentaires
        self.all_preds.extend(predicted.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
    
    def compute(self):
        accuracy = 100 * self.correct / self.total
        avg_loss = self.loss / self.total
        
        # Calcul des métriques additionnelles (optionnel)
        precision = precision_score(self.all_targets, self.all_preds, average='macro', zero_division=0)
        recall = recall_score(self.all_targets, self.all_preds, average='macro', zero_division=0)
        f1 = f1_score(self.all_targets, self.all_preds, average='macro', zero_division=0)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

    def sync_distributed(self, world_size):
        """Pour la synchronisation multi-GPU (sera utilisé par Ahlam)"""
        if not torch.distributed.is_initialized():
            return
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        stats = torch.tensor([self.loss, self.correct, self.total], device=device)
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        
        self.loss = stats[0].item()
        self.correct = int(stats[1].item())
        self.total = int(stats[2].item())