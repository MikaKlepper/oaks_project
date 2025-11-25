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
     # Architecture
    probe_type: str
    hidden_dim: int
    num_layers: int

    # Optimization
    epochs: int
    lr: float
    batch_size: int
    device: str

    optimizer: str
    weight_decay: float
    momentum: float
    loss: str

    def make_loss(self):
        name= self.loss.lower()
        if name == "crossentropy":
            return nn.CrossEntropyLoss()
        elif name == "mse":
            return nn.MSELoss()
        elif name == "bce":
            return nn.BCELoss()
        else:
            raise ValueError(f"Unsupported loss function: {self.loss}")

    def make_optimizer(self, model_params):
        name = self.optimizer.lower()
        if name == "adam":
            return optim.Adam(model_params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "adamw":
            return optim.AdamW(model_params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "rmsprop":
            return optim.RMSprop(model_params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "sgd":
            return optim.SGD(model_params, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        
class TorchProbe(BaseProbe):
    def __init__(self, model: nn.Module, cfg: TorchProbeConfig):
        self_cfg = cfg
        self.model = model.to(self_cfg.device)
        self.criterion = self_cfg.make_loss()
        self.optimizer = self_cfg.make_optimizer(self.model.parameters())
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        device = self.cfg.device
        dataset = TensorDataset(torch.tensor(X, dtype=torch.float32).to(device), torch.tensor(y, dtype=torch.long).to(device))
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)
        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(
            f"[TorchProbe] Training on {device} "
            f"({n_params:,} trainable parameters)"
        )
        logging.info(str(self.model))

        best_loss = float("inf")
        best_state = self.model.state_dict()
        best_epoch = 0

        for epoch in tqdm.tqdm(range(self.cfg.epochs), desc="Training TorchProbe", unit="epoch", leave=True):
            self.model.train()
            running_loss = 0.0

            for xb, yb in dataloader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            logging.info(f"Epoch {epoch+1}/{self.cfg.epochs} - Loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = {k:v.cpu() for k,v in self.model.state_dict().items() if v is not None}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logging.info(
                f"Restored best model from epoch {best_epoch+1} with loss {best_loss:.4f}"
            )





    




# pipeline/probes.py

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sklearn.linear_model import LogisticRegression as SkLogistic
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC, SVC


# ============================================================
# Base Probe Interface
# ============================================================

class BaseProbe(ABC):
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the probe on numpy arrays."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class predictions."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Optional. Only needed for ROC curves.
        Default: not implemented.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support predict_proba()"
        )

    @abstractmethod
    def save(self, path: Union[str, Path]):
        """Save model to disk."""
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]):
        """Load model from disk."""
        pass


# ============================================================
# Torch Classification Heads
# ============================================================

class LinearHead(nn.Module):
    """Simple linear classifier."""
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    """Simple MLP classifier with configurable hidden dim & layers."""
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, num_layers: int):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(max(0, num_layers - 2)):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ============================================================
# TorchProbeConfig — unified hyperparameters
# ============================================================

@dataclass
class TorchProbeConfig:
    # Architecture
    probe_type: str
    hidden_dim: int
    num_layers: int

    # Optimization
    epochs: int
    lr: float
    batch_size: int
    device: str

    optimizer: str = "adam"
    weight_decay: float = 0.0
    momentum: float = 0.9
    loss: str = "crossentropy"

    def make_loss(self):
        name = self.loss.lower()
        if name in ["crossentropy", "ce"]:
            return nn.CrossEntropyLoss()
        if name == "bce":
            return nn.BCEWithLogitsLoss()
        if name == "mse":
            return nn.MSELoss()
        raise ValueError(f"Unknown loss: {self.loss}")

    def make_optimizer(self, params):
        name = self.optimizer.lower()

        if name == "adam":
            return optim.Adam(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "adamw":
            return optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        if name == "sgd":
            return optim.SGD(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        if name == "rmsprop":
            return optim.RMSprop(
                params,
                lr=self.lr,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
            )
        raise ValueError(f"Unknown optimizer: {self.optimizer}")


# ============================================================
# TorchProbe
# ============================================================

class TorchProbe(BaseProbe):
    """Generic wrapper around a torch.nn.Module classifier."""
    def __init__(self, model: nn.Module, cfg: TorchProbeConfig):
        self.model = model.to(cfg.device)
        self.cfg = cfg

        self.criterion = cfg.make_loss()
        self.optimizer = cfg.make_optimizer(self.model.parameters())

    def fit(self, X: np.ndarray, y: np.ndarray):
        device = self.cfg.device

        X_t = torch.tensor(X, dtype=torch.float32).to(device)
        y_t = torch.tensor(y, dtype=torch.long).to(device)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(
            f"[TorchProbe] Training on {device} "
            f"({n_params:,} trainable parameters)"
        )

        best_loss = float("inf")
        best_state = None

        for epoch in tqdm(range(self.cfg.epochs), desc="[TorchProbe]", unit="epoch"):
            self.model.train()
            running_loss = 0.0

            for xb, yb in loader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0)

            epoch_loss = running_loss / len(loader.dataset)
            logging.info(
                f"[TorchProbe] epoch {epoch+1}/{self.cfg.epochs} "
                f"loss={epoch_loss:.4f}"
            )

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logging.info(f"[TorchProbe] Restored best model (loss={best_loss:.4f})")

    def predict(self, X: np.ndarray) -> np.ndarray:
        device = self.cfg.device
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        device = self.cfg.device
        X_t = torch.tensor(X, dtype=torch.float32).to(device)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            probs = torch.softmax(logits, dim=1)

        return probs.cpu().numpy()

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logging.info(f"[TorchProbe] Saved checkpoint to {path}")

    def load(self, path: Union[str, Path]):
        path = Path(path)
        state = torch.load(path, map_location=self.cfg.device)
        self.model.load_state_dict(state)
        logging.info(f"[TorchProbe] Loaded checkpoint from {path}")


# ============================================================
# SklearnProbe
# ============================================================

class SklearnProbe(BaseProbe):
    """Wrap any sklearn classifier to conform to BaseProbe interface."""
    def __init__(self, model):
        self.model = model

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        raise NotImplementedError(
            f"{self.model.__class__.__name__} does not support predict_proba()"
        )

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logging.info(f"[SklearnProbe] Saved checkpoint to {path}")

    def load(self, path: Union[str, Path]):
        path = Path(path)
        self.model = joblib.load(path)
        logging.info(f"[SklearnProbe] Loaded checkpoint from {path}")


# ============================================================
# Helpers
# ============================================================

def default_probe_path(cfg, exp_root: Union[str, Path], is_torch: bool) -> Path:
    """
    Default checkpoint location for a given configuration.
    Saved under experiment_root, per probe type.
    """
    exp_root = Path(exp_root)
    probe_name = str(cfg.probe.type).lower()
    ext = "pt" if is_torch else "joblib"
    return exp_root / f"probe_{probe_name}.{ext}"


# ============================================================
# Probe Factory
# ============================================================

def build_probe(cfg, input_dim: int, num_classes: int) -> BaseProbe:
    """
    Build a fully configured probe from the OmegaConf cfg.

    Supported probe.type:
      - "linear"      -> torch LinearHead
      - "mlp"         -> torch MLPHead
      - "logreg"      -> sklearn LogisticRegression
      - "svm_linear"  -> sklearn LinearSVC
      - "svm_rbf"     -> sklearn SVC (rbf, probability=True)
      - "knn"         -> sklearn KNN (k = fewshot.k or 5)
    """
    probe_type = str(cfg.probe.type).lower()

    # device + runtime hyperparams
    device = getattr(cfg.runtime, "device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    torch_cfg = TorchProbeConfig(
        probe_type=probe_type,
        hidden_dim=getattr(cfg.probe, "hidden_dim", 256) or 256,
        num_layers=getattr(cfg.probe, "layers", 3) or 3,
        epochs=cfg.runtime.epochs,
        lr=cfg.runtime.lr,
        batch_size=cfg.runtime.batch_size,
        device=device,
        optimizer=getattr(cfg.runtime, "optimizer", "adam"),
        weight_decay=getattr(cfg.runtime, "weight_decay", 0.0),
        momentum=getattr(cfg.runtime, "momentum", 0.9),
        loss=getattr(cfg.runtime, "loss", "crossentropy"),
    )

    # Torch probes
    if probe_type == "linear":
        model = LinearHead(input_dim, num_classes)
        return TorchProbe(model, torch_cfg)

    if probe_type == "mlp":
        model = MLPHead(input_dim, torch_cfg.hidden_dim, num_classes, torch_cfg.num_layers)
        return TorchProbe(model, torch_cfg)

    # Sklearn probes
    if probe_type == "logreg":
        model = SkLogistic(max_iter=1000, n_jobs=-1)
        return SklearnProbe(model)

    if probe_type == "svm_linear":
        model = LinearSVC()
        return SklearnProbe(model)

    if probe_type == "svm_rbf":
        model = SVC(kernel="rbf", probability=True)
        return SklearnProbe(model)

    if probe_type == "knn":
        k = getattr(cfg.fewshot, "k", None) or 5
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        return SklearnProbe(model)

    raise ValueError(f"Unknown probe type: {cfg.probe.type}")
