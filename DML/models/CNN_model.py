import torch
import torch.nn as nn
from torchvision import models
import timm
from torchinfo import summary

class EmbeddingModel(nn.Module):
    def __init__(self, num_classes, backbone_name, embedding_size=512):
        super().__init__()
        self.n_classes = num_classes
        self.embedding_size = embedding_size
        
        # Use ResNet152 as the backbone with num_classes=0 to remove classifier head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0  # Remove classifier head
        )
        
        # Get the feature dimension from backbone
        self.in_features = self.backbone.num_features  # Works with num_classes=0
        
        # Add embedding layer
        self.embedding = nn.Sequential(
            nn.Linear(self.in_features, self.embedding_size),
            nn.BatchNorm1d(self.embedding_size),
            nn.ReLU(inplace=True)
        )
        
        # Classifier layer (for regular classification task)
        self.classifier = nn.Linear(self.embedding_size, self.n_classes)

    def forward(self, x, return_embeddings=False):
        features = self.backbone(x)  # Returns a single feature vector
        embeddings = self.embedding(features)
        
        if return_embeddings:
            return embeddings
        
        logits = self.classifier(embeddings)
        return logits, embeddings
    
if __name__=='__main__':
    # Print model architecture
    model = EmbeddingModel(num_classes=3, backbone_name='resnet50')
    # summary(model, (1, 3, 224, 224), device='cpu')
    