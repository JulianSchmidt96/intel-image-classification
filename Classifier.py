import torch
import torch.nn as nn
import torch.nn.functional as F
from ImageClassificationBase import ImageClassificationBase


class Classifier(ImageClassificationBase):
    
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
            
            nn.Flatten(),
            nn.Linear(360000, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,6)
        )
        
        
    def forward(self, xb):
        return self.model(xb)
   