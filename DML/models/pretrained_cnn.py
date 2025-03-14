import torch
import torch.nn as nn
from torchvision import models

class PretrainedCNN(nn.Module):
    def __init__(self, model_name='resnet18', num_classes=3):
        super(PretrainedCNN, self).__init__()
        if model_name == 'resnet18':
            self.model = models.resnet18(pretrained=True)
        elif model_name == 'efficientnet':
            self.model = models.efficientnet_b0(pretrained=True)
        elif model_name == 'mobilenet':
            self.model = models.mobilenet_v2(pretrained=True)
        else:
            raise ValueError("Unsupported model name")

        # Modify the final layer to match the number of classes
        if model_name == 'resnet18':
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        elif model_name == 'efficientnet':
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        elif model_name == 'mobilenet':
            self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)

    def forward(self, x):
        return self.model(x)
