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
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
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
        #elif name == "bce":
            #return nn.BCELoss()
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
        self.cfg = cfg
        self.model = model.to(self.cfg.device)
        self.criterion = self.cfg.make_loss()
        self.optimizer = self.cfg.make_optimizer(self.model.parameters())
    
    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor], dtype=torch.dtype) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(dtype=dtype, device=self.cfg.device)
        return torch.tensor(x, dtype=dtype, device=self.cfg.device)
        
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        device = self.cfg.device
        X_t = self._to_tensor(X, dtype=torch.float32)
        y_t = self._to_tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        dataloader = DataLoader(dataset, batch_size=self.cfg.batch_size, shuffle=True)

        
        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(
            f"[TorchProbe] Training on {device} "
            f"({n_params:,} trainable parameters)"
        )
        logging.info(str(self.model))

        best_loss = float("inf")
        best_state = None
        best_epoch = 0

        for epoch in tqdm(range(self.cfg.epochs), desc="Training TorchProbe", unit="epoch", leave=True):
            self.model.train()
            running_loss = 0.0

            for xb, yb in dataloader:
                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0) # because loss is averaged over batch

            epoch_loss = running_loss / len(dataloader.dataset)
            logging.info(f"Epoch {epoch+1}/{self.cfg.epochs} - Loss: {epoch_loss:.4f}")

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                best_state = {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}

        if best_state is not None:
            self.model.load_state_dict(best_state)
            logging.info(
                f"Restored best model from epoch {best_epoch+1} with loss {best_loss:.4f}"
            )
        
    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        device = self.cfg.device
        X_t = self._to_tensor(X, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            preds = torch.argmax(logits, dim=1)

        return preds.cpu().numpy()

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        device = self.cfg.device
        X_t = self._to_tensor(X, dtype=torch.float32)

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

# sklearn probe wrapper

class SklearnProbe(BaseProbe):
    def __init__(self, model):
        self.model = model

    def _ensure_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x
    
    def fit(self, X: Union[np.ndarray, torch.Tensor], y: Union[np.ndarray, torch.Tensor]):
        logging.info(f"[SklearnProbe] Fitting {self.model.__class__.__name__}")
        self.model.fit(self._ensure_numpy(X), self._ensure_numpy(y))

    def predict(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        return self.model.predict(self._ensure_numpy(X))

    def predict_proba(self, X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(self._ensure_numpy(X))
        else:
            raise NotImplementedError(f"{self.model.__class__.__name__} does not implement predict_proba method.")

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logging.info(f"[SklearnProbe] Saved checkpoint to {path}")

    def load(self, path: Union[str, Path]):
        path = Path(path)
        self.model = joblib.load(path)
        logging.info(f"[SklearnProbe] Loaded checkpoint from {path}")

# helper to get default probe path
def default_probe_path(prepared, exp_root: Union[str, Path], is_torch: bool) -> Path:
    exp_root = Path(exp_root)
    probe_name = str(prepared["probe"]["type"]).lower()
    ext = "pt" if is_torch else "joblib"
    return exp_root / f"probe_{probe_name}.{ext}"
    
def build_probe(prepared, input_dim: int, num_classes: int) -> BaseProbe:
    probe_type = str(prepared["probe"]["type"]).lower()

    # device
    device = prepared["runtime"]["device"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Torch config
    torch_cfg = TorchProbeConfig(
        probe_type=probe_type,
        hidden_dim=prepared["probe"]["hidden_dim"],
        num_layers=prepared["probe"]["num_layers"],
        epochs=prepared["runtime"]["epochs"],
        lr=prepared["runtime"]["lr"],
        batch_size=prepared["runtime"]["batch_size"],
        device=device,
        optimizer=prepared["runtime"]["optimizer"],
        weight_decay=prepared["runtime"]["weight_decay"],
        momentum=prepared["runtime"]["momentum"],
        loss=prepared["runtime"]["loss"],
    )

    # Torch probes
    if probe_type == "linear":
        model = LinearHead(input_dim, num_classes)
        return TorchProbe(model, torch_cfg)

    if probe_type == "mlp":
        model = MLPClassifier(input_dim, torch_cfg.hidden_dim, num_classes, torch_cfg.num_layers)
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
        k = prepared["probe"]["knn_neighbors"]
        model = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        return SklearnProbe(model)

    raise ValueError(f"Unknown probe type: {prepared.probe.type}")
