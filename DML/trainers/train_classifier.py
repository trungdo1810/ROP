import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        CE_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-CE_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * CE_loss
        return focal_loss.mean()

class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, weight=None):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.weight = weight

    def forward(self, inputs, targets):
        return nn.CrossEntropyLoss(weight=self.weight)(inputs, targets)

class ClassifierTrainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load loss configuration
        with open("../../configs/config.yaml") as f:
            config = yaml.safe_load(f)
            
        loss_config = config['loss']
        self.num_classes = loss_config['num_classes']
        
        # Initialize loss function
        if loss_config['type'] == 'focal':
            self.criterion = FocalLoss(
                gamma=loss_config['focal']['gamma'],
                alpha=loss_config['focal']['alpha']
            )
        elif loss_config['type'] == 'balanced_ce':
            self.criterion = BalancedCrossEntropyLoss(
                weight=torch.tensor(loss_config['balanced_ce']['weight'])
            )
        else:  # CE loss
            self.criterion = nn.CrossEntropyLoss(
                weight=torch.tensor(loss_config['ce']['weight'])
            )
            
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=config['training']['learning_rate']
        )

    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        
        for inputs, labels in self.train_loader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)

    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        return val_loss / len(self.val_loader), correct / total

    def train(self, epochs):
        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss, val_acc = self.validate()
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

if __name__ == "__main__":
    # Implement your model and dataloaders here
    # Example skeleton:
    # model = YourModelClass()
    # train_loader = create_train_dataloader()
    # val_loader = create_val_dataloader()
    # trainer = ClassifierTrainer(model, train_loader, val_loader)
    # trainer.train(epochs=50)
    pass
