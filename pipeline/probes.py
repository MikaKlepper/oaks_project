# pipeline/probes.py
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


# asbtract class all probes should inherit from
class BaseProbe(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError(f"{self.__class__.__name__} does not implement predict_proba method.")

    @abstractmethod
    def save(self, path: Union[str, Path]):
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]):
        pass


# torch heads : linear, mlp
class LinearHead(nn.Module):
    """A simple linear classification head."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

class MLPClassifier(nn.Module):
    """
    A simple MLP classifier with a tunable number of hidden layers.
    """
    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
 
 
 # torch probe wrapper
@dataclass
class TorchProbeConfig:
    


