# models/models_.py
# Define simple neural network models for slide-level classification.
import torch.nn as nn

class SlideClassifier(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)
